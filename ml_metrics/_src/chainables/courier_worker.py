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
"""Courier worker that can take and run a registered makeable instance."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from concurrent import futures
import dataclasses as dc
import queue
import random
import threading
import time
from typing import Any, NamedTuple, TypeVar

from absl import logging
from ml_metrics._src import types
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import courier_utils
from ml_metrics._src.utils import iter_utils

_HRTBT_INTERVAL_SECS = 15
_LOGGING_INTERVAL_SEC = 30
_NUM_TOTAL_FAILURES_THRESHOLD = 60
_HRTBT_THRESHOLD_SECS = 180
_T = TypeVar('_T')

Task = courier_utils.Task
GeneratorTask = courier_utils.GeneratorTask


def wait_until_alive(address: str, deadline_secs: float = 120):
  """Wait until the worker at the address is alive."""
  Worker(address).wait_until_alive(deadline_secs=deadline_secs)


def _as_generator_task(
    task: GeneratorTask | types.Resolvable,
) -> GeneratorTask:
  if isinstance(task, GeneratorTask):
    return task
  return GeneratorTask.new(task, courier_method='next_batch_from_generator')


MaybeDoneTasks = NamedTuple(
    'DoneNotDoneTask',
    [
        ('done', list[courier_utils.FutureLike]),
        ('not_done', list[courier_utils.FutureLike]),
    ],
)


def get_results(
    states: Iterable[courier_utils.FutureLike[_T]], timeout: float | None = None
) -> list[_T]:
  """Gets the result of the futures, blocking by default."""
  return [
      lazy_fns.maybe_unpickle(task.result())
      for task in wait(states, timeout=timeout).done
  ]


def get_exceptions(
    states: list[courier_utils.FutureLike], timeout: float | None = None
) -> list[BaseException]:
  """Gets the exceptions from all states, non blocking."""
  return [
      exc
      for state in wait(states, timeout=timeout).done
      if (exc := state.exception()) is not None
  ]


def wait(
    tasks: Iterable[courier_utils.FutureLike], timeout: float | None = None
) -> MaybeDoneTasks:
  """Waits for the tasks to complete optionally with a timeout.

  Args:
    tasks: The tasks to wait for.
    timeout: The timeout in seconds to wait for the tasks to complete.

  Returns:
    A named tuple with the done tasks under 'done' and not done tasks under
    'not_done'.
  """
  start_time = time.time()
  done_tasks, not_done_tasks = [], list(tasks)
  while not_done_tasks:
    still_not_done_tasks = []
    for task in not_done_tasks:
      if task.done():
        done_tasks.append(task)
      else:
        still_not_done_tasks.append(task)
    not_done_tasks = still_not_done_tasks
    if timeout is not None and time.time() - start_time > timeout:
      return MaybeDoneTasks(done_tasks, not_done_tasks)
  return MaybeDoneTasks(done_tasks, not_done_tasks)


def _is_queue_full(e: Exception) -> bool:
  # Courier worker returns a str reprenestation of the exception.
  return isinstance(e, queue.Full) or 'queue.Full' in getattr(e, 'message', '')


def is_timeout(e: Exception | None) -> bool:
  # absl::status::kDeadlineExceeded
  return getattr(e, 'code', 0) == 4


_InputAndFuture = NamedTuple(
    'InputAndFuture', [('input_', Any), ('future', futures.Future[bytes])]
)


@dc.dataclass(frozen=True, kw_only=True, eq=True)
class _WorkerConfig(courier_utils.ClientConfig):

  def make(self) -> Worker:
    return Worker(**dc.asdict(self))


class Worker(courier_utils.CourierClient):
  """Courier client wrapper that works as a chainable worker."""

  def __init__(
      self,
      address: str,
      *,
      call_timeout: float = 0.0,
      max_parallelism: int = 1,
      heartbeat_threshold_secs: float = _HRTBT_THRESHOLD_SECS,
      iterate_batch_size: int = 1,
  ):
    """Initiates a new worker that will connect to a courier server.

    Args:
      address: Address of the worker. If the string does not start with "/" or
        "localhost" then it will be interpreted as a custom BNS registered
        server_name (constructor passed to Server).
      call_timeout: Sets a timeout to apply to the calls. If 0 then no timeout
        is applied.
      max_parallelism: The maximum number of parallel calls to the worker.
      heartbeat_threshold_secs: The threshold to consider the worker alive.
      iterate_batch_size: The batch size to use when iterating an iterator.
    """
    super().__init__(
        address,
        call_timeout=call_timeout,
        max_parallelism=max_parallelism,
        heartbeat_threshold_secs=heartbeat_threshold_secs,
        iterate_batch_size=iterate_batch_size,
    )
    self._worker_pool = None

  @property
  def configs(self) -> _WorkerConfig:
    return _WorkerConfig(
        address=self.address,
        max_parallelism=self.max_parallelism,
        heartbeat_threshold_secs=self.heartbeat_threshold_secs,
        iterate_batch_size=self.iterate_batch_size,
        call_timeout=self._call_timeout,
    )

  @property
  def worker_pool(self) -> WorkerPool | None:
    return self._worker_pool

  # TODO: b/375668959 - Revamp _locker as a normal thread lock and uses
  # worker_pool to indicate the lock owner.
  def is_available(self, worker_pool: WorkerPool | None = None) -> bool:
    """Checks whether the worker is available to the worker pool."""
    return not self._lock.locked() or self._worker_pool is worker_pool

  def is_locked(self, worker_pool: WorkerPool | None = None) -> bool:
    """Checks whether the worker is locked optionally with a worker pool."""
    return self._lock.locked() and (
        not worker_pool or self._worker_pool is worker_pool
    )

  def acquire_by(
      self, worker_pool: WorkerPool, *, blocking: bool = False
  ) -> bool:
    """Acquires the worker if not acquired already."""
    with self._states_lock:
      if self._worker_pool is not worker_pool and self._lock.acquire(
          blocking=blocking
      ):
        self._worker_pool = worker_pool
      return self._worker_pool is worker_pool

  def release(self):
    """Releases the worker."""
    with self._states_lock:
      if self._lock.locked():
        self._lock.release()
      self._worker_pool = None


async def _iterate_until_complete(aiterator, output_queue):
  async for elem in aiterator:
    output_queue.put(elem)


class WorkerPool:
  """Worker group that constructs a group of courier workers.

  Default configs of the workers are set by `Worker()`, and can be overridden
  by explicitly setting non-zero values for the arguments.
  """

  _workers: list[Worker]

  def __init__(
      self,
      names_or_workers: Iterable[str | Worker] = (),
      *,
      call_timeout: float = 0,
      max_parallelism: int = 0,
      heartbeat_threshold_secs: float = 0,
      iterate_batch_size: int = 0,
  ):
    self._workers = []
    for worker in names_or_workers:
      worker = Worker(worker) if isinstance(worker, str) else worker
      self._workers.append(
          Worker(
              worker.address,
              call_timeout=call_timeout or worker.call_timeout,
              max_parallelism=max_parallelism or worker.max_parallelism,
              heartbeat_threshold_secs=heartbeat_threshold_secs
              or worker.heartbeat_threshold_secs,
              iterate_batch_size=iterate_batch_size
              or worker.iterate_batch_size,
          )
      )

  @property
  def server_names(self) -> list[str]:
    return [worker.address for worker in self._workers]

  @property
  def acquired_workers(self) -> list[Worker]:
    return [worker for worker in self._workers if worker.is_locked(self)]

  def _acquire_all(
      self,
      workers: Iterable[Worker] | None = None,
      num_workers: int = 0,
      blocking: bool = False,
  ) -> list[Worker]:
    """Attemps to acquire all workers, returns acquired workers."""
    if workers is None:
      workers = self._workers
    result = []
    for worker in workers:
      if blocking:
        if worker.acquire_by(self, blocking=True):
          result.append(worker)
      elif worker.is_available(self) and worker.acquire_by(self):
        result.append(worker)
      if len(result) == num_workers:
        break
    return result

  def release_all(self, workers: Iterable[Worker] = ()):
    workers = workers or self._workers
    for worker in workers:
      if worker.is_available(self):
        worker.release()

  def wait_until_alive(
      self,
      deadline_secs: float = 180.0,
      *,
      minimum_num_workers: int = 1,
  ):
    """Waits for the workers to be alive with retries."""
    ticker = time.time()
    while (delta_time := time.time() - ticker) < deadline_secs:
      try:
        workers = self.workers
        # Proceed if reached minimum number of workers.
        if len(workers) >= minimum_num_workers:
          logging.info(
              'chainable: %s', f'pool connected {len(workers)} workers'
          )
          return
        logging.log_every_n_seconds(
            logging.INFO,
            f'waiting for workers, connected {len(workers)}/'
            f'{minimum_num_workers} minimum workers',
            _LOGGING_INTERVAL_SEC,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            'chainable: %s', f'exception when connecting: {type(e)}, {e}'
        )
      time.sleep(0)
    unconnected_workers = list(set(self.all_workers) - set(self.workers))
    unconnected_worker_strs = ', '.join(str(w) for w in unconnected_workers[:3])
    raise ValueError(
        f'Failed to connect to {minimum_num_workers=} workers after '
        f'{delta_time:.2f}sec: connected {len(self.workers)} workers. '
        f'First three unconnected servers: {unconnected_worker_strs}'
    )

  def call_and_wait(self, *args, courier_method='maybe_make', **kwargs) -> Any:
    """Calls the workers and waits for the results."""
    self._acquire_all()
    states = [
        c.call(*args, courier_method=courier_method, **kwargs)
        for c in self._workers
    ]
    states = [state for state in states if state is not None]
    try:
      result = get_results(states)
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise e
    finally:
      self.release_all()
    return result

  def set_timeout(self, timeout: int):
    for c in self._workers:
      c.call_timeout = timeout

  # TODO: b/311207032 - deprectate this method in favor of next_idle_worker.
  def idle_workers(self) -> list[Worker]:
    """Attempts to acquire and returns alive workers that have capacity."""
    return [
        worker
        for worker in self._workers
        if worker.is_available(self) and worker.has_capacity and worker.is_alive
    ]

  def next_idle_worker(
      self,
      workers: Iterable[Worker] | None = None,
      *,
      maybe_acquire: bool = False,
  ) -> Worker | None:
    """Non-blocking acquire and yields alive workers that have capacity."""
    workers = self._workers if workers is None else workers
    unacquired_workers: list[Worker] = []
    # Find the worker from acquired worker first.
    for worker in workers:
      if worker.is_locked(self):
        if worker.has_capacity and worker.is_alive:
          return worker
      elif maybe_acquire:
        unacquired_workers.append(worker)
    for worker in unacquired_workers:
      if worker.acquire_by(self) and worker.has_capacity and worker.is_alive:
        return worker

  @property
  def workers(self) -> list[Worker]:
    """Workers that are alive."""
    return [c for c in self._workers if c.is_alive]

  @property
  def all_workers(self) -> list[Worker]:
    """All workers regardless of their status."""
    return self._workers

  @property
  def addresses(self) -> list[str]:
    return [worker.address for worker in self._workers]

  @property
  def num_workers(self):
    return len(self._workers)

  def shutdown(self, blocking: bool = False):
    """Attempting to shut down workers."""
    while blocking and len(self.workers) < self.num_workers:
      remaining = self.num_workers - len(self.workers)
      logging.info(
          'chainable: %s',
          f'shutting down {self.num_workers} workers, remaining'
          f' {remaining} workers are not connected, retrying.',
      )
      time.sleep(6)
    remaining = len(self.all_workers) - len(self.workers)
    logging.info(
        'chainable: %s',
        f'shutting down {len(self.workers)} workers, {remaining=} workers are'
        ' not connected, needs to be manually shutdown.',
    )
    states = [worker.shutdown() for worker in self.all_workers]
    time.sleep(0.1)
    return states

  def run(self, task: lazy_fns.LazyFn | Task) -> Any:
    """Run lazy object or task within the worker pool."""
    self.wait_until_alive()
    worker = None
    start_time = time.time()
    while worker is None:
      worker = self.next_idle_worker(maybe_acquire=True)
      time.sleep(0)
      if time.time() - start_time > 180:
        raise ValueError('No worker is available.')
    # Always set blocking to True as run is blocking.
    task = Task.maybe_as_task(task).set(blocking=True)
    result = worker.submit(task).result()
    worker.release()
    return result

  def iterate(
      self,
      task_iterator: Iterable[GeneratorTask | lazy_fns.LazyFn],
      *,
      generator_result_queue: queue.SimpleQueue[Any],
      retry_threshold: int = _NUM_TOTAL_FAILURES_THRESHOLD,
      total_tasks: int = 0,
  ) -> Iterator[Any]:
    """Iterates through the result of a generator if the iterator task."""
    task_iterator = iter(task_iterator)
    output_queue = queue.SimpleQueue()
    start_time = time.time()
    event_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    loop_thread.start()
    running_tasks: list[GeneratorTask] = []
    failed_tasks: list[GeneratorTask] = []
    tasks: list[GeneratorTask] = []
    timeout_tasks: list[GeneratorTask] = []
    running_workers: set[courier_utils.CourierClient] = set()
    running_total, finished_cnt, timeout_cnt = 0, 0, 0
    batch_cnt = 0

    def iterate():
      nonlocal batch_cnt, running_total, finished_cnt, timeout_cnt
      nonlocal running_tasks, running_workers, timeout_tasks
      exhausted = False
      prev_ticker = start_time
      prev_batch_cnt = batch_cnt
      while not exhausted or tasks or running_tasks:
        workers: list[Worker] = list(set(self.idle_workers()) - running_workers)
        # TODO: b/311207032 - Prefer proven workers before shuffling.
        random.shuffle(workers)
        for worker in workers:
          # Only append new task when existing queue is empty, this is to ensure
          # failed tasks are retried before new tasks are submitted.
          if not tasks and not exhausted:
            try:
              tasks.append(_as_generator_task(next(task_iterator)))
              running_total += 1
            except StopIteration:
              exhausted = True
          if tasks:
            task = tasks.pop().set(worker=worker)
            logging.info(
                'chainable: %s', f'submitting task to worker {worker.address}'
            )
            aiter_until_complete = _iterate_until_complete(
                worker.async_iterate(
                    task, generator_result_queue=generator_result_queue
                ),
                output_queue=output_queue,
            )
            task = task.set(
                state=asyncio.run_coroutine_threadsafe(
                    aiter_until_complete, event_loop
                ),
            )
            running_tasks.append(task)
        while not output_queue.empty():
          batch_cnt += 1
          yield output_queue.get()

        # Acquiring unfinished and failed tasks
        new_failed_tasks, timeout_tasks, still_running_tasks = [], [], []
        for task in running_tasks:
          if task.done():
            if exc := task.exception():
              if isinstance(exc, TimeoutError) or is_timeout(exc):
                logging.warning(
                    'chainable: %s', f'task timeout, worker: {task.worker}'
                )
                timeout_tasks.append(task.set(_exc=None))
              else:
                logging.exception(
                    'chainable: %s',
                    f'task failed with exception: {task.exception()}, {task=}',
                )
                new_failed_tasks.append(task)
            else:
              finished_cnt += 1
          elif task.is_alive:
            still_running_tasks.append(task)
          else:
            logging.warning(
                'chainable: %s', f'worker timeout, worker: {task.worker}'
            )
            timeout_tasks.append(task.set(_exc=None))
        running_tasks = still_running_tasks
        # Preemptively cancel task from the timeout workers.
        for task in timeout_tasks:
          if (state := task.state) is not None:
            state.cancel()
        if new_failed_tasks:
          # Failed tasks likely caused by non-transient error, unretriable.
          logging.error(
              'chainable: %s', f'{len(new_failed_tasks)} new failed tasks.'
          )
          # Return at first failure.
          failed_tasks.extend(new_failed_tasks)
          break
        if timeout_tasks:
          logging.info('chainable: %s', f'{len(timeout_tasks)} tasks Timeout.')
          tasks.extend(timeout_tasks)
          timeout_cnt += len(timeout_tasks)
          if timeout_cnt > retry_threshold:
            break
          logging.info(
              'chainable: %s',
              f'{len(timeout_tasks)} tasks Timeout, retrying: {timeout_tasks}',
          )
        running_workers = {task.worker for task in running_tasks if task.worker}
        if (ticker := time.time()) - prev_ticker > _LOGGING_INTERVAL_SEC:
          delta_time = ticker - prev_ticker
          delta_cnt = batch_cnt - prev_batch_cnt
          failed_cnt = len(failed_tasks)
          total_cnt = total_tasks or running_total
          running_cnt = len(running_tasks)
          logging.info(
              'chainable: %s',
              f'async throughput: {delta_cnt / delta_time:.2f} batches/s;'
              ' progress:'
              f' {running_cnt}/{failed_cnt}/{finished_cnt}/{total_cnt} (running/failed/finished/total)'
              f' with {timeout_cnt} retries in {delta_time:.2f} secs.',
          )
          prev_ticker, prev_batch_cnt = ticker, batch_cnt

    try:
      yield from iterate()
    except KeyboardInterrupt:
      logging.warning('chainable: %s', 'KeyboardInterrupt during iteration.')
    finally:
      # Stop fetching here and prefetching at the workers too.
      for task in running_tasks:
        assert task.state is not None
        if task.worker is not None:
          task.worker.call(courier_method='stop_prefetch')
        task.state.cancel()
      event_loop.call_soon_threadsafe(event_loop.stop)
      while not output_queue.empty():
        batch_cnt += 1
        yield output_queue.get()
      loop_thread.join()
      event_loop.close()
      if generator_result_queue:
        generator_result_queue.put(iter_utils.STOP_ITERATION)
      if failed_tasks:
        raise RuntimeError(
            f'Failed at {running_total}/{total_tasks} task.'
        ) from failed_tasks[-1].exception()
      if timeout_cnt > retry_threshold and timeout_tasks:
        assert (e := timeout_tasks[-1].exception()) is not None
        raise TimeoutError(
            f'Too many Timeouts: {timeout_cnt} > {retry_threshold}, last'
            f' error: {e.args[0]}'
        )
    delta_time = time.time() - start_time
    logging.info(
        'chainable: %s',
        f'iterate finished with {finished_cnt}/{running_total} (finished/total)'
        f' in {delta_time:.2f}s, throughput: {batch_cnt / delta_time:.2f}/s.',
    )
