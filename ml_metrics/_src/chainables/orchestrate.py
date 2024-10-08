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

import asyncio
from collections.abc import Callable, Iterator
from concurrent import futures
import copy
import dataclasses
import queue
import random
import threading
import time
from typing import Any, cast

from absl import logging
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform as transform_lib
from ml_metrics._src.utils import iter_utils

_MASTER = 'master'
_LOGGING_INTERVAL_SECS = 60.0


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
  logging.info('chainable: distributed on %d shards', len(sharded_tasks))
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
        raise ValueError('chainable: no aggregations found in the pipeline.')

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
  logging.info('chainable: iterator: %s', iterator)
  yield from iterator


@dataclasses.dataclass(kw_only=True)
class StageState:
  state: futures.Future[Any]
  result_queue: iter_utils.IteratorQueue[Any]
  name: str = ''
  progress: iter_utils.Progress | None = None


@dataclasses.dataclass(kw_only=True)
class RunnerState:
  """The overall runner state."""

  stages: list[StageState]
  event_loop: asyncio.AbstractEventLoop
  master_server: courier_server.CourierServerWrapper

  @property
  def progress(self) -> iter_utils.Progress | None:
    return self.stages[-1].progress if self.stages else None

  def wait(self, mode=futures.FIRST_EXCEPTION):
    result = futures.wait(
        (s.state for s in self.stages),
        return_when=mode,
    )
    self.event_loop.call_soon_threadsafe(self.event_loop.stop)
    if self.master_server.has_started:
      courier_worker.cached_worker(self.master_server.address).shutdown()
    stage_names = ','.join(f'"{s.name}"' for s in self.stages)
    logging.info('chainable: pipeline with stages %s finished.', stage_names)
    return result

  @property
  def result_queue(self) -> iter_utils.IteratorQueue[Any]:
    return self.stages[-1].result_queue

  def stage_progress(self) -> list[iter_utils.Progress | None]:
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
              f'chainable: stage {i} failed, stage: {s.name}'
          ) from e
    return result


@dataclasses.dataclass(kw_only=True)
class RunnerResource:
  """The resource for the runner."""

  worker_pool: courier_worker.WorkerPool | None = None
  buffer_size: int = 256
  timeout: float | None = None
  num_workers: int = 999999


def _async_run_single_stage(
    transform: transform_lib.TreeTransform,
    *,
    event_loop: asyncio.AbstractEventLoop,
    thread_pool: futures.ThreadPoolExecutor,
    resource: RunnerResource,
    master_server: courier_server.CourierServerWrapper,
    input_queue: iter_utils.IteratorQueue[Any] | None = None,
    ignore_failures: bool = False,
) -> StageState:
  """Asyncronously runs a single stage."""
  # TODO: b/356633410 - Support ignore_failures.
  del ignore_failures
  worker_pool = resource.worker_pool
  if worker_pool and transform.input_iterator is not None:
    raise ValueError(
        'chainable: input_iterator is not supported with worker_pool.'
    )
  if worker_pool and isinstance(transform, transform_lib.AggregateTransform):
    raise ValueError(
        'chainable: AggregateTransform is not supported with worker_pool.'
    )
  result_q = iter_utils.AsyncIteratorQueue(
      resource.buffer_size,
      timeout=resource.timeout,
      name=f'{transform.name}(output)',
  )

  def iterate_with_worker_pool():
    start_time = time.time()
    logging.debug('chainable: "%s" started with worker pool', transform.name)
    assert worker_pool is not None
    input_iterator = None
    if input_queue is not None:
      if not master_server.has_started:
        logging.debug('chainable: starting master %s', master_server.address)
        master_server.start(daemon=True)
      remote_input_q = courier_server.make_remote_queue(
          input_queue,
          server_addr=master_server.address,
          name=f'{transform.name}(input@{master_server.address})',
      )
      input_iterator = remote_input_q
    lazy_iterator = (
        lazy_fns.trace(transform).make(recursive=False).iterate(input_iterator)
    )
    iterating = {}
    min_workers, max_workers = 1, resource.num_workers
    worker_pool.wait_until_alive(deadline_secs=600)
    # Make sure the master is connectable before starting the workers.
    master_server.wait_until_alive(deadline_secs=180)
    ticker = time.time()
    while not result_q.exhausted or iterating:
      # Check the workerpool requirements and set up a new one when needed.
      # No input_queue is considered has input because the transform itself is
      # a datasource operation.
      has_input = input_queue is None or input_queue.qsize()
      num_workers = len(iterating)
      # At least schedule min_workers and maximum max_workers when there are
      # still input left.
      if num_workers < min_workers or (has_input and num_workers < max_workers):
        workers = list(set(worker_pool.workers) - set(iterating))
        random.shuffle(workers)
        worker = worker_pool.next_idle_worker(workers, maybe_acquire=True)
        if worker is not None:
          remote_iterator = worker.async_iter(
              lazy_iterator, name=f'{transform.name}(remote_iter)'
          )
          state = asyncio.run_coroutine_threadsafe(
              result_q.async_enqueue_from_iterator(remote_iterator),
              event_loop,
          )
          iterating[worker] = state
          logging.info('chainable: iterating with %d workers.', len(iterating))

      # Check the states of the workers, release when done or crashed.
      for worker, state in copy.copy(iterating).items():
        if state.done():
          del iterating[worker]
          worker.release()
          if exc := state.exception():
            logging.exception(
                'chainable: worker %s failed with exception: %s, %s',
                worker.server_name,
                type(exc),
                exc,
            )
            raise exc
          logging.info(
              'chainable: worker %s released, remaining %d.',
              worker.server_name,
              len(iterating),
          )
      if time.time() - ticker > _LOGGING_INTERVAL_SECS:
        logging.info(
            'chainable: %s async_iter progress %d and througput %.2f/sec',
            transform.name,
            result_q.progress.cnt,
            result_q.progress.cnt / (time.time() - start_time),
        )
        ticker = time.time()
      time.sleep(0)
    logging.info(
        'chainable: "%s" done with worker pool, average thoughput: %.2f/sec',
        transform.name,
        result_q.progress.cnt / (time.time() - start_time),
    )

  def iterate_in_process():
    start_time = time.time()
    input_iterator = None
    if input_queue is not None:
      input_iterator = iter(input_queue)
    iterator = transform.make(recursive=False).iterate(
        input_iterator=input_iterator
    )
    result_q.enqueue_from_iterator(iterator)
    logging.info(
        'chainable: "%s" done in process, average throughput: %.2f/sec.',
        transform.name,
        result_q.progress.cnt / (time.time() - start_time),
    )

  iter_fn = iterate_with_worker_pool if worker_pool else iterate_in_process
  logging.info(
      'chainable: "%s" started %s.',
      transform.name,
      'with worker pool' if worker_pool else 'in process',
  )
  return StageState(
      state=thread_pool.submit(iter_fn),
      result_queue=result_q,
      name=transform.name,
      progress=result_q.progress,
  )


def run_pipeline_interleaved(
    pipeline: transform_lib.TreeTransform,
    master_server: courier_server.CourierServerWrapper,
    resources: dict[str, RunnerResource] | None = None,
    ignore_failures: bool = False,
) -> RunnerState:
  """Run a pipeline with stages running interleaved."""
  input_queue = None
  resources = resources or {}
  master_server.build_server()
  logging.info('chainable: resolved master address: %s', master_server.address)
  thread_pool = futures.ThreadPoolExecutor()
  event_loop = asyncio.new_event_loop()
  event_loop_thread = threading.Thread(target=event_loop.run_forever)
  event_loop_thread.start()
  stages = []
  for k, p in pipeline.named_transforms().items():
    runner = _async_run_single_stage(
        p,
        thread_pool=thread_pool,
        event_loop=event_loop,
        input_queue=input_queue,
        resource=resources.get(k, RunnerResource()),
        ignore_failures=ignore_failures,
        master_server=master_server,
    )
    stages.append(runner)
    input_queue = runner.result_queue
  return RunnerState(
      stages=stages, event_loop=event_loop, master_server=master_server
  )
