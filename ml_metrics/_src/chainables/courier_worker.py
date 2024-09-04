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

import abc
import asyncio
import collections
from collections.abc import AsyncIterator, Iterable, Iterator
from concurrent import futures
import copy
import dataclasses as dc
import functools
import itertools
import queue
import random
import threading
import time
import typing
from typing import Any, Generic, NamedTuple, Protocol, Self, TypeVar

from absl import logging
import courier
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import func_utils
from ml_metrics._src.utils import iter_utils
import more_itertools as mit


_LOGGING_INTERVAL_SEC = 30
_NUM_TOTAL_FAILURES_THRESHOLD = 60
_T = TypeVar('_T')


@functools.lru_cache
def _cached_client(addr: str, call_timeout: int):
  return courier.Client(addr, call_timeout=call_timeout)


@func_utils.cache_without_kwargs
def _cached_worker(
    addr: str,
    *,
    call_timeout: int = 60,
    max_parallelism: int = 1,
    heartbeat_threshold_secs: int = 180,
    iterate_batch_size: int = 1,
):
  return Worker(
      addr,
      call_timeout=call_timeout,
      max_parallelism=max_parallelism,
      heartbeat_threshold_secs=heartbeat_threshold_secs,
      iterate_batch_size=iterate_batch_size,
  )


class _FutureLike(Protocol[_T]):

  @abc.abstractmethod
  def done(self) -> bool:
    """Indicates the future is done."""

  @abc.abstractmethod
  def result(self) -> _T:
    """Returns the result of the future."""

  @abc.abstractmethod
  def exception(self) -> BaseException | None:
    """Returns the exception of the future."""


# TODO: b/311207032 - Implements Future interface for Task.
@dc.dataclass(kw_only=True, frozen=True)
class Task(_FutureLike[_T]):
  """Lazy function that runs on courier methods.

  Attributes:
    args: The positional arguments later used to pass in the fn.
    kwargs: The named arguments later used to pass in the fn.
    blocking: Whether the wait for the call to complete.
    keep_result: Whether to keep the result.
    worker: The courier worker that runs this task.
    parent_task: The parent task that has to be run first.
    state: the result of the task.
    courier_method: the courier method of the task.
    exception: the exception of the running this task if there is any.
    result: get the result of the task if there is any.
    server_name: The server address this task is sent to.
    is_alive: True if the worker running this task is alive.
  """

  args: tuple[Any, ...] = ()
  kwargs: dict[str, Any] = dc.field(default_factory=dict)
  blocking: bool = False
  worker: Worker | None = None
  parent_task: 'Task | None' = None
  state: futures.Future[Any] | None = None
  courier_method: str = 'maybe_make'

  @classmethod
  def new(
      cls,
      *args,
      blocking: bool = False,
      courier_method: str = 'maybe_make',
      **kwargs,
  ) -> Self:
    """Convenient function to make a Task."""
    return cls(
        args=args,
        kwargs=kwargs,
        blocking=blocking,
        courier_method=courier_method,
    )

  @classmethod
  def from_list_of_tasks(cls, tasks: list[Self]) -> Self:
    iter_tasks = iter(tasks)
    task = next(iter_tasks)
    assert isinstance(task, Task)
    for next_task in iter_tasks:
      task = dc.replace(next_task, parent_task=task)
    return task

  # The followings are to implement the _FutureLike interfaces.
  def done(self) -> bool:
    """Checks whether the task is done."""
    if (state := self.state) is not None:
      return state.done()
    return False

  def result(self) -> _T:
    if self.state is not None:
      return lazy_fns.pickler.loads(self.state.result())

  def exception(self) -> BaseException | None:
    if (state := self.state) is not None:
      return state.exception()

  @property
  def server_name(self) -> str:
    assert self.worker is not None
    return self.worker.server_name

  @property
  def is_alive(self) -> bool:
    worker = self.worker
    assert worker is not None
    return worker.is_alive

  def set(self, **kwargs) -> Self:
    return dc.replace(self, **kwargs)

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
      result = dc.replace(each_task, parent_task=result)
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


@dc.dataclass(kw_only=True, frozen=True)
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


def _as_task(task: Task | lazy_fns.LazyFn) -> Task:
  return Task.new(task) if isinstance(task, lazy_fns.LazyFn) else task


def _as_generator_task(task: GeneratorTask | lazy_fns.LazyFn) -> GeneratorTask:
  return GeneratorTask.new(task) if isinstance(task, lazy_fns.LazyFn) else task


MaybeDoneTasks = NamedTuple(
    'DoneNotDoneTask',
    [('done', list[_FutureLike]), ('not_done', list[_FutureLike])],
)


def _maybe_unpickle(value: Any) -> Any:
  if isinstance(value, bytes):
    return lazy_fns.pickler.loads(value)
  return value


def get_results(
    states: Iterable[_FutureLike[_T]], timeout: float | None = None
) -> list[_T]:
  """Gets the result of the futures, blocking by default."""
  return [
      _maybe_unpickle(task.result())
      for task in wait(states, timeout=timeout).done
  ]


def get_exceptions(
    states: list[_FutureLike], timeout: float | None = None
) -> list[BaseException]:
  """Gets the exceptions from all states, non blocking."""
  return [
      exc
      for state in wait(states, timeout=timeout).done
      if (exc := state.exception()) is not None
  ]


def wait(
    tasks: Iterable[_FutureLike], timeout: float | None = None
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


@dc.dataclass(frozen=True)
class RemoteObject(Generic[_T], lazy_fns.Resolvable[lazy_fns.LazyObject[_T]]):
  """Remote object holds remote reference that behaves like a local object."""
  value: lazy_fns.LazyObject[_T]
  worker_addr: str = dc.field(kw_only=True)

  @classmethod
  def new(cls, value, *, worker: str | Worker) -> Self:
    if not isinstance(value, lazy_fns.LazyObject):
      value = lazy_fns.LazyObject.new(value)
    worker_addr = worker.server_name if isinstance(worker, Worker) else worker
    return cls(value, worker_addr=worker_addr)

  def __hash__(self) -> int:
    return hash(self.value)

  def __eq__(self, other: Self) -> bool:
    return self.value == other.value

  @property
  def id(self):
    assert (v := self.value) is not None
    return v.id

  @property
  def worker(self) -> Worker:
    return _cached_worker(self.worker_addr)

  def deref_(self) -> futures.Future[Any]:
    return self.worker.call(self.value)

  def result_(self) -> Any:
    return lazy_fns.pickler.loads(self.deref_().result())

  # Overrides to support pickling when getattr is overridden.
  def __getstate__(self):
    return dict(self.__dict__)

  # Overrides to support pickling when getattr is overridden.
  def __setstate__(self, state):
    self.__dict__.update(state)

  # The following are remote builtin methods.
  def __call__(self, *args, **kwargs) -> Self:
    """Calling a LazyFn records a lazy result of the call."""
    return RemoteObject.new(
        self.value(*args, **kwargs), worker=self.worker_addr
    )

  def __getattr__(self, name) -> Self:
    return RemoteObject.new(getattr(self.value, name), worker=self.worker_addr)

  def __getitem__(self, key) -> Self:
    return RemoteObject.new(self.value[key], worker=self.worker_addr)


def _is_queue_full(e: Exception) -> bool:
  # Courier worker returns a str reprenestation of the exception.
  return 'queue.Full' in getattr(e, 'message', '')


def _is_courier_timeout(exc: Exception | None) -> bool:
  # absl::status::kDeadlineExceeded
  return getattr(exc, 'code', 0) == 4


_InputAndFuture = collections.namedtuple(
    '_InputAndFuture', ['input_', 'future']
)


@dc.dataclass(slots=True)
class RemoteQueues:
  """Combine multiple remote queues into one as iterators when en/dequeueing."""

  remote_queues: set[RemoteObject[queue.Queue[Any]]] = dc.field(
      default_factory=set
  )
  timeout_secs: int | None = None

  def __post_init__(self):
    self.remote_queues = set(self.remote_queues)

  def add(self, remote_queue: RemoteObject[queue.Queue[Any]]):
    self.remote_queues.add(remote_queue)

  def remove(self, remote_queue: RemoteObject[queue.Queue[Any]]):
    self.remote_queues.remove(remote_queue)

  def dequeue(self) -> Iterator[Any]:
    """Roundrobin across all queues and get the next avaialble item."""
    states = {}
    reamining_queues = copy.copy(self.remote_queues)
    ticker = time.time()
    while reamining_queues:
      for remote_queue in tuple(reamining_queues):
        if remote_queue not in states:
          states[remote_queue] = remote_queue.get_nowait().deref_()
        if (state := states[remote_queue]).done():
          try:
            result = lazy_fns.pickler.loads(state.result())
            del states[remote_queue]
            if isinstance(result, StopIteration):
              reamining_queues.remove(remote_queue)
            else:
              yield result
              ticker = time.time()
          except Exception as e:  # pylint: disable=broad-exception-caught
            # Courier worker returns a str reprenestation of the exception.
            if 'queue.Empty' in getattr(e, 'message', ''):
              if (
                  self.timeout_secs is not None
                  and time.time() - ticker > self.timeout_secs
              ):
                raise TimeoutError(
                    f'Dequeue timeout after {self.timeout_secs}s.'
                ) from e
              continue
            else:
              raise e
      time.sleep(0)

  def enqueue(self, values: Iterable[Any]):
    """Roundrobin across all queues and put the item to a available queue."""
    values = iter(values)
    enqueueing: dict[RemoteObject[queue.Queue[Any]], _InputAndFuture] = {}
    preferred_queues = copy.copy(self.remote_queues)
    if not self.remote_queues:
      raise queue.Full(f'No remote queue: {self.remote_queues=}')
    ticker = time.time()
    stopping = False
    while not stopping or enqueueing:
      # Reset the state of the queues that has finished enqueing.
      for remote_queue, (input_, state) in tuple(enqueueing.items()):
        if state.done():
          try:
            assert lazy_fns.pickler.loads(state.result()) is None
            ticker = time.time()
            preferred_queues.add(remote_queue)
            del enqueueing[remote_queue]
            # if not iter_utils.is_stop_iteration(input_):
            #   yield input_
          except Exception as e:  # pylint: disable=broad-exception-caught
            preferred_queues.discard(remote_queue)
            if _is_queue_full(e) or _is_courier_timeout(e):
              if time.time() - ticker > self.timeout_secs:
                raise TimeoutError(
                    f'Enqueue timeout after {self.timeout_secs}s.'
                ) from e
              # For StopIteration, retry on the same queue.
              if iter_utils.is_stop_iteration(input_):
                enqueueing[remote_queue] = _InputAndFuture(
                    input_, remote_queue.put_nowait(input_).deref_()
                )
              else:
                values = mit.prepend(input_, values)
            else:
              raise e

      # Pull the next value from the iterator.
      value = next(values, iter_utils.STOP_ITERATION)
      if iter_utils.is_stop_iteration(value):
        # Only starts to put StopIteration when all values are enqueued. This is
        # to ensure retrying is still possible.
        if not stopping and not enqueueing:
          # Broadcast the StopIteration to all queues.
          for remote_queue in self.remote_queues:
            enqueueing[remote_queue] = _InputAndFuture(
                value, remote_queue.put_nowait(value).deref_()
            )
          stopping = True
      else:
        # Prefer to send item to queues that haven't thrown exception yet.
        available_queues = preferred_queues - set(enqueueing)
        if not available_queues:
          available_queues = self.remote_queues - set(enqueueing)
        # Enqueue the value to a random available queue.
        if available_queues:
          remote_queue = random.choice(list(available_queues))
          enqueueing[remote_queue] = _InputAndFuture(
              value, remote_queue.put_nowait(value).deref_()
          )
        else:
          values = mit.prepend(value, values)
      time.sleep(0)


def _normalize_args(args, kwargs):
  """Normalizes the args and kwargs to be picklable."""
  result_args = []
  for arg in args:
    try:
      result_args.append(lazy_fns.pickler.dumps(arg))
    except Exception as e:
      raise ValueError(f'Having issue pickling arg: {arg}') from e
  result_kwargs = {}
  for k, v in kwargs.items():
    try:
      result_kwargs[k] = lazy_fns.pickler.dumps(v)
    except Exception as e:
      raise ValueError(f'Having issue pickling {k}: {v}') from e
  return result_args, result_kwargs


# TODO(b/311207032): Adds unit test to cover logic for disconneted worker.
class Worker:
  """Courier client wrapper that works as a chainable worker."""

  server_name: str
  max_parallelism: int = 1
  heartbeat_threshold_secs: float = 180.0
  iterate_batch_size: int = 1
  _call_timeout: float = 60.0
  _lock: threading.Lock = threading.Lock()
  _worker_pool: WorkerPool | None
  _shutdown_requested: bool = False
  _client: courier.Client
  _pendings: list[futures.Future[Any]]
  _heartbeat_client: courier.Client
  _heartbeat: futures.Future[Any] | None = dc.field(default=None, init=False)
  _last_heartbeat: float

  def __init__(
      self,
      server_name: str | None = '',
      *,
      call_timeout: float = 60,
      max_parallelism: int = 1,
      heartbeat_threshold_secs: float = 180,
      iterate_batch_size: int = 1,
  ):
    self.server_name = server_name or ''
    self.max_parallelism = max_parallelism
    self.heartbeat_threshold_secs = heartbeat_threshold_secs
    self.iterate_batch_size = iterate_batch_size
    self._call_timeout = call_timeout
    self._lock = threading.Lock()
    self._worker_pool = None
    self._shutdown_requested = True if server_name is None else False
    self._pendings = []
    self._client = _cached_client(self.server_name, call_timeout=call_timeout)
    self._heartbeat_client = _cached_client(
        self.server_name, call_timeout=self.heartbeat_threshold_secs
    )
    self._heartbeat = None
    self._last_heartbeat = 0.0

  @property
  def call_timeout(self) -> float:
    return self._call_timeout

  @property
  def worker_pool(self) -> WorkerPool | None:
    return self._worker_pool

  def __hash__(self):
    assert self.server_name, 'server_name must be set for hashing.'
    return hash(self.server_name)

  def __eq__(self, other):
    return self.server_name == other.server_name

  @call_timeout.setter
  def call_timeout(self, call_timeout: float):
    self.set_timeout(call_timeout)

  def set_timeout(self, call_timeout: float):
    self._call_timeout = call_timeout
    self._client = _cached_client(
        self.server_name, call_timeout=self.call_timeout
    )

  def is_available(self, worker_pool: WorkerPool) -> bool:
    """Checks whether the worker is available to the worker pool."""
    return not self._lock.locked() or (self._worker_pool is worker_pool)

  def is_locked(self, worker_pool: WorkerPool | None = None) -> bool:
    """Checks whether the worker is locked optionally with a worker pool."""
    return self._lock.locked() and (
        not worker_pool or self._worker_pool is worker_pool
    )

  def acquire_by(
      self, worker_pool: WorkerPool, *, blocking: bool = False
  ) -> bool:
    """Acquires the worker if not acquired already."""
    if self._worker_pool is not worker_pool and self._lock.acquire(
        blocking=blocking
    ):
      self._worker_pool = worker_pool
    return self._worker_pool is worker_pool

  def release(self):
    """Releases the worker."""
    if self._lock.locked():
      self._lock.release()
    self._worker_pool = None

  @property
  def has_capacity(self) -> bool:
    return len(self.pendings) < self.max_parallelism

  def _check_heartbeat(self) -> bool:
    """Ping the worker to check the heartbeat once."""
    if not self._heartbeat:
      self._heartbeat = self._heartbeat_client.futures.heartbeat()
    try:
      if self._heartbeat.done():
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
      sleep_interval_secs: float = 1.0,
  ):
    """Waits for the worker to be alive with retries."""
    sleep_interval_secs = max(sleep_interval_secs, 0.1)
    num_attempts = int(max(deadline_secs // sleep_interval_secs, 1))
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

  def submit(self, task: Task[_T] | lazy_fns.Resolvable[_T]) -> Task[_T]:
    """Runs tasks sequentially and returns the task."""
    if isinstance(task, lazy_fns.Resolvable):
      task = Task.new(task)
    result = []
    for task in task.flatten():
      while not self.has_capacity:
        time.sleep(0)
      state = self.call(
          *task.args, courier_method=task.courier_method, **task.kwargs
      )
      if task.blocking:
        futures.wait([state])
      result.append(task.set(state=state, worker=self))
    return Task.from_list_of_tasks(result)

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
      self.submit(task.parent_task)
    # Artificially insert a pending state to block other tasks.
    generator_state = futures.Future()
    self._pendings.append(generator_state)
    batch_cnt = 0
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
        assert isinstance(output_batch, list)
        if output_batch:
          if iter_utils.is_stop_iteration(stop_iteration := output_batch[-1]):
            exhausted = True
            generator_result_queue.put(stop_iteration.value)
            output_batch = output_batch[:-1]
          elif isinstance(exc := output_batch[-1], Exception):
            if exc != ValueError('generator already executing'):
              raise output_batch[-1]
        for elem in output_batch:
          yield elem
          batch_cnt += 1
      logging.info(
          'chainables: worker %s generator exhausted after %d batches',
          self.server_name,
          batch_cnt,
      )
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

  def cache_info(self):
    return lazy_fns.pickler.loads(
        self.call(courier_method='cache_info').result()
    )

  def shutdown(self) -> futures.Future[Any]:
    self._shutdown_requested = True
    self.state = self._client.futures.shutdown()
    return self.state


class WorkerPool:
  """Worker group that constructs a group of courier workers."""

  _workers: list[Worker]

  def __init__(
      self,
      names_or_workers: Iterable[str] | Iterable[Worker] = (),
      *,
      call_timeout: float = 0,
      max_parallelism: int = 0,
      heartbeat_threshold_secs: int = 0,
      iterate_batch_size: int = 0,
  ):
    if all(isinstance(name, str) for name in names_or_workers):
      self._workers = [_cached_worker(name) for name in names_or_workers]
    elif all(isinstance(worker, Worker) for worker in names_or_workers):
      self._workers = typing.cast(list[Worker], list(names_or_workers))
    else:
      raise TypeError(
          'Expected either a list of names or a list of workers, got'
          f' {names_or_workers}'
      )
    for worker in self._workers:
      worker.call_timeout = call_timeout or worker.call_timeout
      worker.max_parallelism = max_parallelism or worker.max_parallelism
      worker.heartbeat_threshold_secs = (
          heartbeat_threshold_secs or worker.heartbeat_threshold_secs
      )
      worker.iterate_batch_size = (
          iterate_batch_size or worker.iterate_batch_size
      )

  @property
  def server_names(self) -> list[str]:
    return [worker.server_name for worker in self._workers]

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

  def _release_all(self, workers: Iterable[Worker] = ()):
    workers = workers or self._workers
    for worker in workers:
      if worker.is_available(self):
        worker.release()

  def wait_until_alive(
      self,
      deadline_secs: float = 180.0,
      *,
      minimum_num_workers: int = 1,
      sleep_interval_secs: float = 1.0,
  ):
    """Waits for the workers to be alive with retries."""
    sleep_interval_secs = max(sleep_interval_secs, 0.1)
    num_attempts = int(max(deadline_secs // sleep_interval_secs, 1))
    for _ in range(num_attempts):
      try:
        workers = self.workers
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

  def call_and_wait(self, *args, courier_method='', **kwargs) -> Any:
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
      self._release_all()
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
  def num_workers(self):
    return len(self.server_names)

  def shutdown(self, blocking: bool = False):
    """Attemping to shut down workers."""
    while blocking and len(self.workers) < self.num_workers:
      logging.info(
          'chainables: shutting down %d workers, remaining %d workers are not'
          ' connected, retrying.',
          self.num_workers,
          self.num_workers - len(self.workers),
      )
      time.sleep(6)
    logging.info(
        'chainables: shutting down %d workers, remaining %d workers are not'
        ' connected, needs to be manually shutdown.',
        len(self.workers),
        len(self.all_workers) - len(self.workers),
    )
    states = [worker.shutdown() for worker in self.all_workers]
    time.sleep(0.1)
    return states

  def as_completed(
      self,
      task_iterator: Iterable[Task | lazy_fns.LazyObject],
      ignore_failures: bool = False,
  ) -> Iterator[Task]:
    """Run tasks within the worker pool."""
    task_iterator = iter(task_iterator)
    running_tasks: list[Task] = []
    tasks: list[Task] = []
    preferred_workers = set()
    exhausted = False
    while not exhausted or tasks or running_tasks:
      # Submitting the next batch of tasks.
      if not self.workers:
        raise TimeoutError('All workers timeout, check worker status.')
      workers = itertools.chain(
          preferred_workers, set(self.workers) - preferred_workers
      )
      while (tasks or not exhausted) and (
          worker := self.next_idle_worker(workers, maybe_acquire=True)
      ):
        # Ensure failed tasks are retried before new tasks are submitted.
        if not tasks and not exhausted:
          try:
            tasks.append(_as_task(next(task_iterator)))
          except StopIteration:
            exhausted = True
        if tasks:
          running_tasks.append(worker.submit(tasks.pop()))

      # Check the results of the running tasks and retry timeout tasks.
      still_running: list[Task] = []
      for task in running_tasks:
        if task.done():
          if exc := task.exception():
            preferred_workers.discard(task.worker)
            if _is_courier_timeout(exc):
              logging.warning(
                  'chainables: deadline exceeded at %s.',
                  task.server_name,
              )
              tasks.append(task)
            elif ignore_failures:
              logging.exception(
                  'chainables: task failed with exception: %s, task: %s',
                  exc,
                  task.set(parent_task=None),
              )
            else:
              raise exc
          else:
            preferred_workers.add(task.worker)
            yield task
        elif not task.is_alive:
          logging.warning(
              'chainables: Worker %s disconnected.',
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
        running_workers = set(task.worker for task in running_tasks)
        # Reserve same amount of workers as the number of unproven workers.
        num_reserved_workers = len(running_workers - preferred_workers)
        acquired_workers = set(self.acquired_workers)
        reserved_workers = set()
        # Reserve some workers from the preferred workers first.
        if candidate_workers := list(preferred_workers - running_workers):
          reserved_workers.update(
              random.sample(candidate_workers, k=num_reserved_workers)
          )
        elif candidate_workers := list(acquired_workers - running_workers):
          reserved_workers.update(
              random.sample(candidate_workers, k=num_reserved_workers)
          )
        unused_workers = acquired_workers - running_workers - reserved_workers
        self._release_all(unused_workers)
      time.sleep(0.0)
    self._release_all()

  def run(
      self,
      fns: Iterable[lazy_fns.LazyFn | Task] | lazy_fns.LazyFn | Task,
      ignore_failures: bool = False,
  ) -> Any:
    """Run lazy functions or task within the worker pool."""
    if signle_task := isinstance(fns, (lazy_fns.LazyFn, Task)):
      fns = [fns]
    results = get_results(self.as_completed(fns, ignore_failures))
    return results[0] if signle_task else results

  # TODO: b/311207032 - Provide gradual worker lock and unlock mechanism.
  def iterate(
      self,
      task_iterator: Iterable[GeneratorTask | lazy_fns.LazyFn],
      *,
      generator_result_queue: queue.SimpleQueue[Any],
      num_total_failures_threshold: int = _NUM_TOTAL_FAILURES_THRESHOLD,
  ) -> Iterator[Any]:
    """Iterates through the result of a generator if the iterator task."""
    task_iterator = iter(task_iterator)
    output_queue = queue.SimpleQueue()
    start_time = prev_ticker = time.time()
    event_loop = asyncio.new_event_loop()

    async def iterate_until_complete(aiterator, output_queue):
      async for elem in aiterator:
        output_queue.put(elem)

    loop_thread = threading.Thread(target=event_loop.run_forever)
    loop_thread.start()
    running_tasks: list[GeneratorTask] = []
    tasks: list[GeneratorTask] = []
    running_workers: set[Worker] = set()
    total_tasks, finished_tasks_cnt, total_failures_cnt = 0, 0, 0
    batch_cnt, prev_batch_cnt = 0, 0
    exhausted = False
    while not exhausted or tasks or running_tasks:
      workers = list(set(self.idle_workers()) - running_workers)
      # TODO: b/311207032 - Prefer proven workers before shuffling.
      random.shuffle(workers)
      for worker in workers:
        # Only append new task when existing queue is empty, this is to ensure
        # failed tasks are retried before new tasks are submitted.
        if not tasks and not exhausted:
          try:
            tasks.append(_as_generator_task(next(task_iterator)))
            total_tasks += 1
          except StopIteration:
            exhausted = True
        if tasks:
          task = tasks.pop().set(worker=worker)
          logging.info(
              'chainables: submitting task to worker %s', worker.server_name
          )
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
        if task.done():
          if exc := task.exception():
            if _is_courier_timeout(exc):
              disconnected_tasks.append(task)
            else:
              logging.exception(
                  'chainables: task failed with exception: %s, task: %s',
                  task.exception(),
                  task.set(parent_task=None),
              )
              failed_tasks.append(task)
          else:
            finished_tasks_cnt += 1
        else:
          if task.is_alive:
            still_running_tasks.append(task)
          else:
            disconnected_tasks.append(task)
      running_tasks = still_running_tasks

      # Preemptively cancel task from the disconnected workers.
      for task in disconnected_tasks:
        if (state := task.state) is not None:
          state.cancel()
      if failed_tasks or disconnected_tasks:
        logging.info(
            'chainables: %d tasks failed, %d tasks disconnected.',
            len(failed_tasks),
            len(disconnected_tasks),
        )
        tasks.extend(disconnected_tasks)
        total_failures_cnt += len(failed_tasks)
        if total_failures_cnt > num_total_failures_threshold:
          logging.exception(
              'chainables: too many failures: %d > %d, stopping, '
              'last exception:\n %s.',
              total_failures_cnt,
              num_total_failures_threshold,
              failed_tasks[-1].exception(),
          )
          raise failed_tasks[-1].exception()
        else:
          tasks.extend(failed_tasks)
          logging.info(
              'chainables: %d tasks failed, re-trying: %s.',
              len(failed_tasks),
              failed_tasks,
          )
      running_workers = {task.worker for task in running_tasks if task.worker}

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
    assert exhausted and total_tasks == finished_tasks_cnt
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
      generator_result_queue.put(iter_utils.STOP_ITERATION)
