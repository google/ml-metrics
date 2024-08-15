# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Orchestration for the remote distributed chainable workers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent import futures
import dataclasses
import queue
import threading
import time
from typing import Any, cast

from absl import logging
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform as transform_lib
from ml_metrics._src.utils import iter_utils


def sharded_pipelines_as_iterator(
    worker_pool: courier_worker.WorkerPool,
    /,
    define_pipeline: Callable[..., transform_lib.TreeTransform],
    *pipeline_args,
    with_batch_output: bool = True,
    result_queue: (
        queue.SimpleQueue[transform_lib.AggregateResult] | None
    ) = None,
    retry_failures: bool = True,
    **pipeline_kwargs,
) -> Iterator[Any]:
  """The orchestration for the remote distributed chainable workers.

  Args:
    worker_pool: The worker pool to run the pipeline.
    define_pipeline: The pipeline definition.
    *pipeline_args: The pipeline args.
    with_batch_output: Whether to return the batch output.
    result_queue: The queue to output the result.
    retry_failures: Whether to retry when seeing failures.
    **pipeline_kwargs: The pipeline kwargs.

  Yields:
    The batch outputs of of the pipeline.
  """
  calculate_agg_result = result_queue is not None
  num_shards = worker_pool.num_workers
  worker_pool.wait_until_alive(deadline_secs=600)
  sharded_tasks = [
      courier_worker.GeneratorTask.new(
          lazy_fns.trace(define_pipeline)(
              *pipeline_args,
              shard_index=i,
              num_shards=num_shards,
              **pipeline_kwargs,
          )
          .make()
          .iterate(
              with_result=with_batch_output,
              with_agg_state=calculate_agg_result,
              # In the distributed fashion, the result for each shard is not
              # meaningful, thus is always disabled.
              with_agg_result=False,
          ),
      )
      for i in range(num_shards)
  ]
  logging.info('chainables: distributed on %d shards', len(sharded_tasks))
  # Inference and aggregations.
  states_queue = queue.SimpleQueue()
  if calculate_agg_result:

    def compute_result(
        states_queue: queue.SimpleQueue[
            transform_lib.AggregateResult | StopIteration
        ],
        result_queue: queue.SimpleQueue[transform_lib.AggregateResult],
    ):
      agg_fn = define_pipeline(
          *pipeline_args,
          shard_index=0,
          num_shards=num_shards,
          **pipeline_kwargs,
      ).make(mode=transform_lib.RunnerMode.AGGREGATE)
      if not agg_fn.has_agg:
        raise ValueError('chainables: no aggregations found in the pipeline.')

      def iterate_agg_state():
        while True:
          try:
            state = states_queue.get()
          except queue.Empty:
            time.sleep(0)
            continue
          if iter_utils.is_stop_iteration(state):
            return
          yield cast(transform_lib.AggregateResult, state).agg_state

      merged_state = agg_fn.merge_states(iterate_agg_state())
      # At most only one item in the output_q.
      result_queue.put(
          transform_lib.AggregateResult(
              agg_state=merged_state,
              agg_result=agg_fn.get_result(merged_state),
          ),
      )

    thread = threading.Thread(
        target=compute_result, args=(states_queue, result_queue)
    )
    thread.start()

  # Does not allow 50% of total number of tasks failed.
  faiure_threshold = int(len(sharded_tasks) * 0.5) + 1 if retry_failures else 0
  iterator = worker_pool.iterate(
      sharded_tasks,
      generator_result_queue=states_queue,
      num_total_failures_threshold=faiure_threshold,
  )
  logging.info('chainables: iterator: %s', iterator)
  yield from iterator


@dataclasses.dataclass(kw_only=True)
class StageState:
  state: futures.Future[Any]
  result_queue: queue.Queue[Any]
  name: str = ''
  _progress: list[int]

  @property
  def progress(self) -> int:
    return self._progress[0] if self._progress else -1


@dataclasses.dataclass(kw_only=True)
class RunnerState:
  """The overall runner state."""

  stages: list[StageState]

  @property
  def progress(self) -> int:
    return self.stages[-1].progress if self.stages else -1

  def wait(self, mode=futures.FIRST_EXCEPTION):
    return futures.wait(
        (s.state for s in self.stages),
        return_when=mode,
    )

  def iterate(self) -> Iterator[Any]:
    return iter_utils.dequeue_as_generator(self.stages[-1].result_queue)

  def stage_progress(self) -> list[int]:
    return [s.progress for s in self.stages]

  def done(self):
    return all(s.state.done() for s in self.stages)

  def exception(self):
    results = []
    for i, s in enumerate(self.stages):
      if s.state.done() and (exc := s.state.exception()):
        results.append((i, exc))
    return results

  def wait_and_maybe_raise(self):
    result = self.wait()
    for i, s in enumerate(self.stages):
      if s.state.done() and s.state.exception():
        try:
          _ = s.state.result()
        except Exception as e:  # pylint: disable=broad-exception-caught
          raise ValueError(
              f'chainables: stage {i} failed, stage: {s.name}'
          ) from e
    return result


@dataclasses.dataclass(kw_only=True)
class RunnerResource:
  """The resource for the runner."""

  worker_pool: courier_worker.WorkerPool | None = None
  buffer_size: int = 0


def _async_run_single_stage(
    transform: transform_lib.TreeTransform,
    *,
    thread_pool: futures.ThreadPoolExecutor,
    resource: RunnerResource,
    input_queue: queue.Queue[Any] | None = None,
    ignore_failures: bool = False,
) -> StageState:
  """Asyncronously runs a single stage."""
  worker_pool = resource.worker_pool
  if worker_pool and transform.input_iterator is not None:
    raise ValueError(
        'chainables: input_iterator is not supported with worker_pool.'
    )
  if worker_pool and isinstance(transform, transform_lib.AggregateTransform):
    raise ValueError(
        'chainables: AggregateTransform is not supported with worker_pool.'
    )
  result_q = queue.Queue(maxsize=resource.buffer_size)
  input_iterator = None
  if input_queue is not None:
    input_iterator = iter_utils.dequeue_as_generator(input_queue)
  progress = [0]

  def _iterate_with_worker_pool():
    assert worker_pool is not None
    completed_tasks = worker_pool.as_completed(
        (
            lazy_fns.trace_object(transform).make(recursive=False)(batch)
            for batch in input_iterator
        ),
        ignore_failures=ignore_failures,
    )
    for i, _ in enumerate(
        iter_utils.enqueue_from_generator(
            (task.result() for task in completed_tasks), result_q
        )
    ):
      progress[0] = i + 1

  def _iterate_in_process():
    for i, _ in enumerate(
        iter_utils.enqueue_from_generator(
            transform.make(recursive=False).iterate(
                input_iterator=input_iterator
            ),
            result_q,
        )
    ):
      progress[0] = i

  iterate_fn = _iterate_with_worker_pool if worker_pool else _iterate_in_process
  return StageState(
      state=thread_pool.submit(iterate_fn),
      result_queue=result_q,
      name=transform.name,
      _progress=progress,
  )


def run_pipeline_interleaved(
    pipeline: transform_lib.TreeTransform,
    resources: dict[str, RunnerResource] | None = None,
    ignore_failures: bool = False,
) -> RunnerState:
  """Run a pipeline with stages running interleaved."""
  input_queue = None
  resources = resources or {}
  thread_pool = futures.ThreadPoolExecutor()
  stages = []
  for k, p in pipeline.named_transforms().items():
    runner = _async_run_single_stage(
        p,
        thread_pool=thread_pool,
        input_queue=input_queue,
        resource=resources.get(k, RunnerResource()),
        ignore_failures=ignore_failures,
    )
    stages.append(runner)
    input_queue = runner.result_queue
  return RunnerState(stages=stages)
