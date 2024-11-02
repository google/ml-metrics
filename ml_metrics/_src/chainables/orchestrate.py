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
from collections.abc import Callable, Iterable, Iterator
from concurrent import futures
import copy
import dataclasses
import itertools
import queue
import random
import threading
import time
from typing import Any, cast

from absl import logging
from ml_metrics._src import base_types
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
  fn: Callable[..., Any]
  result_queue: iter_utils.IteratorQueue[Any]
  state: futures.Future[Any] = futures.Future()
  name: str = ''
  progress: iter_utils.Progress | None = None


@dataclasses.dataclass(kw_only=True)
class RunnerState:
  """The overall runner state."""

  stages: list[StageState]
  event_loop: asyncio.AbstractEventLoop
  event_loop_thread: threading.Thread
  master_server: courier_server.CourierServerWrapper
  thread_pool: futures.ThreadPoolExecutor

  @property
  def progress(self) -> iter_utils.Progress | None:
    return self.stages[-1].progress if self.stages else None

  def __enter__(self):
    self.run()
    return self

  def __exit__(self, *args):
    self.wait_and_maybe_raise()

  def run(self):
    for stage in self.stages:
      stage.state = self.thread_pool.submit(stage.fn)

  def wait(self, mode=futures.FIRST_EXCEPTION):
    result = futures.wait(
        (s.state for s in self.stages),
        return_when=mode,
    )
    self.event_loop.call_soon_threadsafe(self.event_loop.stop)
    if self.master_server.has_started:
      self.master_server.stop()
    self.event_loop_thread.join()
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
    """Wait for the pipeline to finish and raise the first exception."""
    result = self.wait()
    for i, s in enumerate(self.stages):
      if s.state.done() and s.state.exception():
        try:
          _ = s.state.result()
        except Exception as e:  # pylint: disable=broad-exception-caught
          raise ValueError(
              f'chainable: stage {i} failed, stage: {s.name}'
          ) from e
    self.thread_pool.shutdown()
    if t := self.master_server.stop():
      t.join()
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
  result_q = iter_utils.AsyncIteratorQueue(
      resource.buffer_size,
      timeout=resource.timeout,
      name=f'{transform.name}(output)',
      thread_pool=thread_pool,
  )

  def iterate_with_worker_pool():
    start_time = time.time()
    logging.info('chainable: "%s" started with worker pool', transform.name)
    assert worker_pool is not None
    input_iterator = None
    if input_queue is not None:
      if not master_server.has_started:
        logging.debug('chainable: starting master %s', master_server.address)
        master_server.start(daemon=True)
      remote_input_q = courier_worker.RemoteIteratorQueue.new(
          input_queue,
          server_addr=master_server.address,
          name=f'{transform.name}(input@{master_server.address})',
      )
      input_iterator = remote_input_q
    lazy_iterator = (
        lazy_fns.trace(transform)
        .make(recursive=False)
        .iterate(input_iterator, with_agg_result=False)
    )
    iterating = {}
    min_workers, max_workers = 1, resource.num_workers
    worker_pool.wait_until_alive(deadline_secs=600)
    # Make sure the master is connectable before starting the workers.
    courier_worker.wait_until_alive(master_server.address, deadline_secs=180)
    ticker = time.time()
    while not result_q.enqueue_done or iterating:
      # Check the workerpool requirements and set up a new one when needed.
      num_workers = len(iterating)
      # At least schedule min_workers and maximum max_workers when there are
      # still input left.
      still_need_workers = (input_queue and num_workers < max_workers)
      if num_workers < min_workers or still_need_workers:
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
                worker,
                type(exc),
                exc,
            )
            raise exc
          logging.info(
              'chainable: worker %s released, remains %d, "%s" enqueue_done=%s',
              worker,
              len(iterating),
              result_q.name,
              result_q.enqueue_done,
          )
      if time.time() - ticker > _LOGGING_INTERVAL_SECS:
        logging.info(
            'chainable: %s async_iter processed %d and througput %.2f/sec',
            transform.name,
            result_q.progress.cnt,
            result_q.progress.cnt / (time.time() - start_time),
        )
        ticker = time.time()
      time.sleep(0)

    # Merge the intermediate aggregation states as aggregation result if there
    # are any.
    if result_q.returned:
      agg_states = []
      for agg_result in result_q.returned:
        assert isinstance(agg_result, transform_lib.AggregateResult)
        if agg_result.agg_state is not None:
          agg_states.append(agg_result.agg_state)
      agg_fn = transform.make(mode=transform_lib.RunnerMode.AGGREGATE)
      logging.debug(
          'chainable: %s merging %d agg states.',
          transform.name,
          len(agg_states),
      )
      agg_state = agg_fn.merge_states(agg_states)
      agg_result = agg_fn.get_result(agg_state)
      result_q.returned.clear()
      result_q.returned.append(
          transform_lib.AggregateResult(
              agg_state=agg_state, agg_result=agg_result
          )
      )
      logging.debug(
          'chainable: %s merged %d agg states.',
          transform.name,
          len(agg_states),
      )

    delta_time = time.time() - start_time
    logging.info(
        'chainable: "%s" done (remote) in %d secs, throughput: %.2f/sec',
        transform.name,
        delta_time,
        result_q.progress.cnt / delta_time,
    )

  def iterate_in_process():
    logging.info('chainable: "%s" started in process', transform.name)
    start_time = time.time()
    input_iterator = None
    if input_queue is not None:
      input_iterator = iter(input_queue)
    iterator = transform.make(recursive=False).iterate(
        input_iterator=input_iterator
    )
    result_q.enqueue_from_iterator(iterator)
    delta_time = time.time() - start_time
    logging.info(
        'chainable: "%s" done (local) in %d secs, throughput: %.2f/sec.',
        transform.name,
        delta_time,
        result_q.progress.cnt / delta_time,
    )

  iter_fn = iterate_with_worker_pool if worker_pool else iterate_in_process
  return StageState(
      fn=iter_fn,
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
      stages=stages,
      event_loop=event_loop,
      event_loop_thread=event_loop_thread,
      master_server=master_server,
      thread_pool=thread_pool,
  )


def as_completed(
    worker_pool: courier_worker.WorkerPool,
    task_iterator: Iterable[courier_worker.Task | base_types.Resolvable],
    ignore_failures: bool = False,
) -> Iterator[Any]:
  """Run tasks within the worker pool."""
  task_iterator = iter(task_iterator)
  running_tasks: list[courier_worker.Task] = []
  tasks: list[courier_worker.Task] = []
  preferred = set()
  exhausted = False
  while not exhausted or tasks or running_tasks:
    # Submitting the next batch of tasks.
    if not worker_pool.workers:
      raise TimeoutError('All workers timeout, check worker status.')
    backup_workers = list(set(worker_pool.workers) - preferred)
    random.shuffle(backup_workers)
    workers = list(itertools.chain(preferred, backup_workers))
    while (tasks or not exhausted) and (
        worker := worker_pool.next_idle_worker(workers, maybe_acquire=True)
    ):
      # Ensure failed tasks are retried before new tasks are submitted.
      if not tasks and not exhausted:
        try:
          tasks.append(courier_worker.Task.maybe_as_task(next(task_iterator)))
        except StopIteration:
          exhausted = True
      if tasks:
        running_tasks.append(worker.submit(tasks.pop()))

    # Check the results of the running tasks and retry timeout tasks.
    still_running: list[courier_worker.Task] = []
    for task in running_tasks:
      if task.done():
        if exc := task.exception():
          preferred.discard(task.worker)
          if isinstance(exc, TimeoutError) or courier_worker.is_timeout(exc):
            logging.warning(
                'chainable: deadline exceeded at %s, retrying task.',
                task.server_name,
            )
            tasks.append(task)
          elif ignore_failures:
            logging.exception(
                'chainable: task failed with exception: %s, task: %s', exc, task
            )
          else:
            raise exc
        else:
          preferred.add(task.worker)
          yield task.result()
      elif not task.is_alive:
        logging.warning(
            'chainable: Worker %s disconnected.',
            task.server_name,
        )
        assert task.state is not None
        task.state.set_exception(TimeoutError(f'{task.server_name} timeout.'))
        tasks.append(task)
      else:
        still_running.append(task)
    running_tasks = still_running

    # Releasing unused workers.
    if exhausted and not tasks:
      running = set(task.worker for task in running_tasks)
      acquired = set(worker_pool.acquired_workers)
      reserved = set()
      # Reserve some workers from the preferred workers first.
      if candidates := list(preferred - running or acquired - running):
        # Reserve same amount of workers as the number of unproven workers.
        num_reserved_workers = len(running - preferred)
        reserved.update(random.sample(candidates, k=num_reserved_workers))
      unused_workers = acquired - running - reserved
      worker_pool.release_all(unused_workers)
    time.sleep(0.0)
  worker_pool.release_all()
