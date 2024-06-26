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
from collections.abc import AsyncIterator, Iterable, Iterator
from concurrent import futures
import dataclasses
import queue
import threading
import time
from typing import Any
from typing import Self

from absl import logging
import courier
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform


_LOGGING_INTERVAL_SEC = 30
_NUM_TOTAL_FAILURES_THRESHOLD = 60
picklers = lazy_fns.picklers


@dataclasses.dataclass(kw_only=True, frozen=True)
class Task:
  """Lazy function that runs on courier methods.

  Attributes:
    args: The positional arguments later used to pass in the fn.
    kwargs: The named arguments later used to pass in the fn.
    blocking: Whether the wait for the call to complete.
    keep_result: Whether to keep the result.
    server_name: The server address this task is sent to.
    parent_task: The parent task that has to be run first.
    state: the result of the task.
    exception: the exception of the running this task if there is any.
    result: get the result of the task if there is any.
  """

  args: tuple[Any, ...] = ()
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  blocking: bool = False
  server_name: str = ''
  parent_task: 'Task | None' = None
  state: futures.Future[Any] | None = None

  @classmethod
  def new(
      cls,
      *args,
      blocking: bool = False,
      **kwargs,
  ):
    """Convenient function to make a Task."""
    return cls(
        args=args,
        kwargs=kwargs,
        blocking=blocking,
    )

  @classmethod
  def from_list_of_tasks(cls, tasks: list['Task']) -> 'Task':
    iter_tasks = iter(tasks)
    task = next(iter_tasks)
    assert isinstance(task, Task)
    for next_task in iter_tasks:
      task = dataclasses.replace(next_task, parent_task=task)
    return task

  @property
  def done(self) -> bool | None:
    """Checks whether the task is done."""
    if self.state is not None:
      return self.state.done()

  @property
  def exception(self):
    return self.state and self.state.exception()

  @property
  def result(self):
    if self.state is None:
      return None
    return picklers.default.loads(self.state.result())

  def set(self, **kwargs) -> Self:
    return dataclasses.replace(self, **kwargs)

  def add_task(
      self,
      task: Task | Any,
      *,
      blocking: bool = False,
  ):
    """Append a task behind this task."""
    if not isinstance(task, Task):
      task = Task.new(
          task,
          blocking=blocking,
      )
    result = self
    for each_task in task.flatten():
      result = dataclasses.replace(each_task, parent_task=result)
    return result

  def add_generator_task(
      self,
      task: GeneratorTask | Any,
      *,
      blocking: bool = False,
  ):
    """Append a task behind this task."""
    if not isinstance(task, GeneratorTask):
      task = GeneratorTask.new(
          task,
          blocking=blocking,
      )
    return self.add_task(task)

  def flatten(self) -> list['Task']:
    if self.parent_task is None:
      return [self]
    return self.parent_task.flatten() + [self]


@dataclasses.dataclass(kw_only=True, frozen=True)
class GeneratorTask(Task):
  """Courier worker communication for generator.

  Attributes:
    args: The positional arguments later used to pass in the fn.
    kwargs: The named arguments later used to pass in the fn.
    blocking: Whether the wait for the call to complete.
    keep_result: Whether to keep the result.
    server_name: The server address this task is sent to.
    parent_task: The parent task that has to be run first.
    state: the result of the task.
    exception: the exception of the running this task if there is any.
    result: get the result of the task if there is any.
  """

  def next_batch(self, worker_pool: WorkerPool):
    return self.set(
        state=worker_pool.get_worker_by_name(
            self.server_name
        ).next_batch_from_generator()
    )


def get_results(
    states: list[futures.Future[Any]],
    with_states: bool = False,
    blocking: bool = True,
):
  """Gets the result of the futures, optionally outputs the non-done states."""
  states_left, result = [], []
  if blocking:
    wait_until_done(states)
  for state in states:
    assert isinstance(
        state, futures.Future
    ), f'state has to be a futures.Future, got {type(state)}'
    if state.done():
      result.append(state.result())
    else:
      states_left.append(state)
  result = [
      picklers.default.loads(state) if isinstance(state, bytes) else state
      for state in result
  ]
  return (result, states_left) if with_states else result


def _get_exception(state: futures.Future[Any]):
  """Gets the exception, non blocking."""
  if state.done():
    return state.exception()


def get_exceptions(
    states: list[futures.Future[Any]],
) -> list[Exception]:
  """Gets the exceptions from all states, non blocking."""
  result = [_get_exception(state) for state in states]
  return [state for state in result if state is not None]


def wait_until_done(states: list[futures.Future[Any]]):
  while not all(state.done() for state in states):
    time.sleep(0.01)


def _normalize_args(args, kwargs):
  """Normalizes the args and kwargs to be picklable."""
  result_args = []
  for arg in args:
    try:
      result_args.append(picklers.default.dumps(arg))
    except Exception as e:
      raise ValueError(f'Having issue pickling arg: {arg}') from e
  result_kwargs = {}
  for k, v in kwargs.items():
    try:
      result_kwargs[k] = picklers.default.dumps(v)
    except Exception as e:
      raise ValueError(f'Having issue pickling {k}: {v}') from e
  return result_args, result_kwargs


# TODO(b/311207032): Adds unit test to cover logic for disconneted worker.
@dataclasses.dataclass
class Worker:
  """Courier client wrapper that works as a chainable worker."""

  server_name: str
  call_timeout: int = 60
  max_parallelism: int = 1
  heartbeat_threshold_secs: int = 180
  iterate_batch_size: int = 1
  _shutdown_requested: bool = False
  _client: courier.Client | None = dataclasses.field(default=None, init=False)
  _pendings: list[futures.Future[Any]] = dataclasses.field(
      default_factory=list, init=False
  )
  _heartbeat_client: courier.Client | None = dataclasses.field(
      default=None, init=False
  )
  _heartbeat: futures.Future[Any] | None = dataclasses.field(
      default=None, init=False
  )
  _last_heartbeat: float = 0.0

  def __post_init__(self):
    self._client = courier.Client(
        self.server_name, call_timeout=self.call_timeout
    )
    self._heartbeat_client = courier.Client(
        self.server_name, call_timeout=self.heartbeat_threshold_secs
    )

  @property
  def has_capacity(self) -> bool:
    return len(self.pendings) < self.max_parallelism

  def _check_heartbeat(self) -> bool:
    """Ping the worker to check the heartbeat once."""
    if not self._heartbeat:
      self._heartbeat = self._heartbeat_client.futures.maybe_make(None)
    try:
      if self._heartbeat.done() and self._heartbeat.result():
        self._heartbeat = None
        self._last_heartbeat = time.time()
        return True
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning(
          'chainables: Worker %s missed a heartbeat.', self.server_name
      )
      self._heartbeat = None
    return False

  def wait_until_alive(
      self,
      deadline_secs: int = 180,
      *,
      sleep_interval_secs: int = 6,
  ):
    """Waits for the worker to be alive with retries."""
    num_attempts = max(deadline_secs // sleep_interval_secs, 1)
    for _ in range(num_attempts):
      try:
        if self.is_alive:
          break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('chainables: exception when connecting: %s', e)
      time.sleep(sleep_interval_secs)
    else:
      raise ValueError(
          f'Failed to connect to worker {self.server_name} after'
          f' {num_attempts} tries.'
      )

  @property
  def is_alive(self) -> bool:
    """Checks whether the worker is alive."""
    if self._shutdown_requested:
      return False
    if self._check_heartbeat():
      return True
    # No last heartbeat recorded, consider it dead.
    if self._last_heartbeat:
      time_passed = time.time() - self._last_heartbeat
      return time_passed < self.heartbeat_threshold_secs
    else:
      return False

  @property
  def pendings(self) -> list[futures.Future[Any]]:
    """Returns the states that are not done."""
    self._pendings = [state for state in self._pendings if not state.done()]
    return self._pendings

  def call(
      self, *args, courier_method: str = '', **kwargs
  ) -> futures.Future[Any]:
    args, kwargs = _normalize_args(args, kwargs)
    courier_method = courier_method or 'maybe_make'
    assert self._client is not None
    state = getattr(self._client.futures, courier_method)(*args, **kwargs)
    self._pendings.append(state)
    return state

  def run_task(self, task: Task) -> Task:
    """Runs tasks sequentially and returns the futures."""
    result = []
    for task in task.flatten():
      while not self.has_capacity:
        time.sleep(0)
      state = self.call(*task.args, **task.kwargs)
      if task.blocking:
        wait_until_done([state])
      result.append(task.set(state=state, server_name=self.server_name))
    return Task.from_list_of_tasks(result)

  def init_generator(self, task: GeneratorTask) -> GeneratorTask:
    parent_task = None
    if task.parent_task is not None:
      parent_task = self.run_task(task.parent_task)
    return task.set(
        server_name=self.server_name,
        state=self.call(
            *task.args, courier_method='init_generator', **task.kwargs
        ),
        parent_task=parent_task,
    )

  def next_batch_from_generator(
      self, batch_size: int = 0
  ) -> futures.Future[Any]:
    batch_size = batch_size or self.iterate_batch_size
    return self.call(
        courier_method='next_batch_from_generator', batch_size=batch_size
    )

  def next_from_generator(self) -> futures.Future[Any]:
    return self.call(courier_method='next_from_generator')

  async def async_iterate(
      self,
      task: GeneratorTask,
      *,
      generator_result_queue: queue.SimpleQueue[transform.AggregateResult],
  ) -> AsyncIterator[Any]:
    """Iterates the generator task."""
    # Make the actual generator.
    if task.parent_task is not None:
      self.run_task(task.parent_task)
    # Artificially insert a pending state to block other tasks.
    generator_state = futures.Future()
    self._pendings.append(generator_state)
    try:
      init_state = self.call(
          *task.args, courier_method='init_generator', **task.kwargs
      )
      assert await asyncio.wrap_future(init_state) is None
      exhausted = False
      while not exhausted:
        output_state = self.next_batch_from_generator(self.iterate_batch_size)
        output_batch = lazy_fns.maybe_make(
            await asyncio.wrap_future(output_state)
        )
        if output_batch:
          if transform.is_stop_iteration(stop_iteration := output_batch[-1]):
            logging.info(
                'chainables: worker %s generator exhausted with retruned: %s',
                self.server_name,
                stop_iteration.value is not None,
            )
            exhausted = True
            print('<<< stop iteration: sending:', stop_iteration.value)
            generator_result_queue.put(stop_iteration.value)
            output_batch = output_batch[:-1]
          elif isinstance(exc := output_batch[-1], Exception):
            if exc != ValueError('generator already executing'):
              raise output_batch[-1]
        for elem in output_batch:
          yield elem
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          'chainables: exception when iterating task: %s',
          task.set(parent_task=None),
      )
      raise e
    finally:
      generator_state.cancel()

  def clear_cache(self):
    return self.call(courier_method='clear_cache')

  def set_timeout(self, call_timeout: int):
    self.call_timeout = call_timeout
    self._client = courier.Client(self.server_name, call_timeout=call_timeout)

  def shutdown(self) -> futures.Future[Any]:
    self._shutdown_requested = True
    self.state = self._client.futures.shutdown()
    return self.state


def _raise_if_return_error(task: Task):
  if not (result := task.state) or (result := task.exception):
    raise TypeError(
        f'Expected iterator, got {result} from'
        f' task: {dataclasses.replace(task, parent_task=None)}'
    )


@dataclasses.dataclass
class WorkerPool:
  """Worker group that constructs a group of courier workers."""

  server_names: list[str]
  call_timeout: int = 30
  max_parallelism: int = 1
  heartbeat_threshold_secs: int = 180
  iterate_batch_size: int = 1
  _workers: dict[str, Worker] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    self._workers = {
        name: Worker(
            name,
            call_timeout=self.call_timeout,
            max_parallelism=self.max_parallelism,
            heartbeat_threshold_secs=self.heartbeat_threshold_secs,
            iterate_batch_size=self.iterate_batch_size,
        )
        for name in self.server_names
    }

  def wait_until_alive(
      self,
      deadline_secs: int = 180,
      *,
      minimum_num_workers: int = 1,
  ):
    """Waits for the workers to be alive with retries."""
    sleep_interval_secs: int = 6
    num_attempts = max(deadline_secs // sleep_interval_secs, 1)
    for _ in range(num_attempts):
      try:
        workers = self.idle_workers()
        logging.info('chainables: Available workers: %d', len(workers))
        # Proceed if reached minimum number of workers.
        if len(workers) >= minimum_num_workers:
          break
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('chainables: exception when connecting: %s', e)
      time.sleep(sleep_interval_secs)
    else:
      raise ValueError(
          f'Failed to connect to workers after {num_attempts} tries: workers:'
          f' {self.server_names}'
      )

  def call_and_wait(self, *args, courier_method='', **kwargs):
    states = [
        c.call(*args, courier_method=courier_method, **kwargs)
        for c in self._workers.values()
    ]
    states = [state for state in states if state is not None]
    return get_results(states)

  def set_timeout(self, timeout: int):
    self.call_timeout = timeout
    for c in self._workers.values():
      c.set_timeout(timeout)

  def idle_workers(self):
    return [c for c in self._workers.values() if c.has_capacity and c.is_alive]

  @property
  def workers(self):
    return [c for c in self._workers.values() if c.is_alive]

  @property
  def num_workers(self):
    return len(self.server_names)

  def get_worker_by_name(self, name: str) -> Worker:
    return self._workers[name]

  def shutdown(self):
    remaining_workers = self.idle_workers()
    logging.info(
        'chainables: shutting down %d workers, remaining %d workers are not'
        ' connected, needs to be manually shutdown.',
        len(remaining_workers),
        len(self.workers) - len(remaining_workers),
    )
    states = [worker.shutdown() for worker in remaining_workers]
    time.sleep(0.1)
    return states

  def run_tasks(
      self, tasks: list[Task], sleep_interval: float = 0.01
  ) -> Iterator[Task]:
    """Run tasks within the worker pool."""
    running_tasks = []
    while tasks or running_tasks:
      if not self.workers:
        raise ValueError(
            'No workers are alive, remaining'
            f' {len(tasks)+len(running_tasks)} tasks.'
        )
      for worker in self.idle_workers():
        if tasks:
          running_tasks.append(worker.run_task(tasks.pop()))
      still_running = []
      for task in running_tasks:
        if task.done:
          yield task
        elif not self._workers[task.server_name].is_alive:
          logging.warning(
              'chainables: Worker %s is not alive, re-appending task %s',
              task.server_name,
              dataclasses.replace(task, parent_task=None),
          )
          tasks.append(task)
        else:
          still_running.append(task)
      running_tasks = still_running
      time.sleep(sleep_interval)

  # TODO: b/311207032 - deprecate run_and_iterat in favor or iterate.
  def run_and_iterate(
      self,
      tasks: Iterable[GeneratorTask],
      num_total_failures_threshold: int = _NUM_TOTAL_FAILURES_THRESHOLD,
  ) -> Iterator[Any]:
    """Iterates through the result of a generator if the iterator task."""
    tasks = list(tasks)
    pending_tasks, running_tasks = [], []
    total_tasks = len(tasks)
    total_failures_cnt, finished_tasks_cnt = 0, 0
    batch_cnt, prev_batch_cnt = 0, 0
    self.wait_until_alive()
    start_time = prev_ticker = time.time()
    while tasks or pending_tasks or running_tasks:
      # Assign to the iterator tasks
      iterating_servers = {
          task.server_name for task in running_tasks + pending_tasks
      }
      for worker in self.idle_workers():
        # Only assign non-running tasks to the non-iterating workers.
        if worker.server_name not in iterating_servers:
          if tasks:
            task = worker.init_generator(tasks.pop())
            pending_tasks.append(task)
      # Check the instantiated iterator, then assign iteratate if successful.
      new_pending_tasks: list[GeneratorTask] = []
      for task in pending_tasks:
        if task.done:
          _raise_if_return_error(task)
          running_tasks.append(task.next_batch(self))
        else:
          new_pending_tasks.append(task)
      pending_tasks = new_pending_tasks
      # Fetching finsihed outputs. Re-collect task if failed during iteration.
      still_running: list[GeneratorTask] = []
      for task in running_tasks:
        if task.done:
          states = task.result
          if states and isinstance(states[-1], Exception):
            batch_cnt += len(states[:-1])
            yield from states[:-1]
            try:
              if isinstance((stop_iteration := states[-1]), StopIteration):
                logging.info(
                    'chainables: worker %s generator exhausted.',
                    task.server_name,
                )
                if stop_iteration.value is not None:
                  yield stop_iteration.value
                finished_tasks_cnt += 1
              else:
                raise states[-1]
            except Exception:  # pylint: disable=broad-exception-caught
              logging.exception(
                  'chainables: exception when iterating, re-appending task %s,'
                  ' \n worker: %s.',
                  dataclasses.replace(task, parent_task=None),
                  task.server_name,
              )
              total_failures_cnt += 1
              if total_failures_cnt > num_total_failures_threshold:
                raise ValueError(
                    f'chainables: too many failures: {total_failures_cnt} >'
                    f' {num_total_failures_threshold}, stopping the iteration.'
                ) from states[-1]
              tasks.append(task)
          else:
            batch_cnt += len(states)
            yield from states
            still_running.append(task.next_batch(self))
        elif not self.get_worker_by_name(task.server_name).is_alive:
          logging.warning(
              'chainables: Worker %s is not alive, re-appending task %s',
              task.server_name,
              dataclasses.replace(task, parent_task=None),
          )
          tasks.append(task)
        else:
          still_running.append(task)
      running_tasks = still_running
      assert (
          finished_tasks_cnt
          + len(running_tasks)
          + len(pending_tasks)
          + len(tasks)
      ) == total_tasks, 'Total tasks mismatch.'
      if time.time() - prev_ticker > _LOGGING_INTERVAL_SEC:
        logging.info(
            'chainables: iterate throughput: %.2f; progress: %d/%d/%d/%d in'
            ' %.2f secs. (running/remaining/finished/total).',
            (batch_cnt - prev_batch_cnt) / (time.time() - prev_ticker),
            len(running_tasks) + len(pending_tasks),
            len(tasks),
            finished_tasks_cnt,
            total_tasks,
            time.time() - prev_ticker,
        )
        prev_ticker = time.time()
        prev_batch_cnt = batch_cnt
      time.sleep(0)
    logging.info(
        'chainables: finished iterating %d/%d in %.2f secs.',
        finished_tasks_cnt,
        total_tasks,
        time.time() - start_time,
    )

  def iterate(
      self,
      tasks: Iterable[GeneratorTask],
      *,
      generator_result_queue: queue.SimpleQueue[Any],
      num_total_failures_threshold: int = _NUM_TOTAL_FAILURES_THRESHOLD,
  ) -> Iterator[Any]:
    """Iterates through the result of a generator if the iterator task."""
    tasks = list(reversed(list(tasks)))
    total_tasks = len(tasks)
    output_queue = queue.SimpleQueue()
    start_time = prev_ticker = time.time()
    event_loop = asyncio.new_event_loop()

    async def iterate_until_complete(aiterator, output_queue):
      async for elem in aiterator:
        output_queue.put(elem)

    loop_thread = threading.Thread(target=event_loop.run_forever)
    loop_thread.start()
    running_tasks: list[GeneratorTask] = []
    iterating_servers = set()
    finished_tasks_cnt, total_failures_cnt = 0, 0
    batch_cnt, prev_batch_cnt = 0, 0
    while tasks or running_tasks:
      workers = [
          worker
          for worker in self.idle_workers()
          if worker.server_name not in iterating_servers
      ]
      for worker in workers:
        if tasks:
          logging.info(
              'chainables: submitting task to worker %s', worker.server_name
          )
          task = tasks.pop().set(server_name=worker.server_name)
          aiter_until_complete = iterate_until_complete(
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
      failed_tasks, disconnected_tasks, still_running_tasks = [], [], []
      for task in running_tasks:
        if task.done:
          if task.exception:
            failed_tasks.append(task)
          else:
            finished_tasks_cnt += 1
        else:
          if self.get_worker_by_name(task.server_name).is_alive:
            still_running_tasks.append(task)
          else:
            disconnected_tasks.append(task)
      running_tasks = still_running_tasks

      # Preemptively cancel task from the disconnected workers.
      for task in disconnected_tasks:
        assert task.state is not None
        task.state.cancel()
      if failed_tasks or disconnected_tasks:
        logging.info(
            'chainables: %d tasks failed, %d tasks disconnected.',
            len(failed_tasks),
            len(disconnected_tasks),
        )
        tasks.extend(failed_tasks + disconnected_tasks)
        total_failures_cnt += len(failed_tasks)
        if total_failures_cnt > num_total_failures_threshold:
          raise ValueError(
              f'chainables: too many failures: {total_failures_cnt} >'
              f' {num_total_failures_threshold}, stopping the iteration.'
          )
        logging.info(
            'chainables: %d tasks failed, re-trying: %s.',
            len(failed_tasks),
            failed_tasks,
        )
      iterating_servers = {task.server_name for task in running_tasks}

      if (ticker := time.time()) - prev_ticker > _LOGGING_INTERVAL_SEC:
        logging.info(
            'chainables: async throughput: %.2f batches/s; progress:'
            ' %d/%d/%d/%d (running/remaining/finished/total) with %d retries in'
            ' %.2f secs.',
            (batch_cnt - prev_batch_cnt) / (ticker - prev_ticker),
            len(running_tasks),
            len(tasks),
            finished_tasks_cnt,
            total_tasks,
            total_failures_cnt,
            ticker - prev_ticker,
        )
        prev_ticker, prev_batch_cnt = ticker, batch_cnt

    assert not running_tasks
    assert total_tasks == finished_tasks_cnt
    event_loop.call_soon_threadsafe(event_loop.stop)
    loop_thread.join()
    while not output_queue.empty():
      batch_cnt += 1
      yield output_queue.get()
    logging.info(
        'chainalbes: finished running with %d/%d (finished/total) in %.2f'
        ' secs, average throughput: %.2f/sec.',
        finished_tasks_cnt,
        total_tasks,
        time.time() - start_time,
        batch_cnt / (time.time() - start_time),
    )
    if generator_result_queue:
      generator_result_queue.put(StopIteration())
