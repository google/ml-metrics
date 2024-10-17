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
from collections.abc import AsyncIterator, Iterable, Iterator
from concurrent import futures
import dataclasses as dc
import functools
import queue
import random
import threading
import time
import typing
from typing import Any, Generic, NamedTuple, Protocol, Self, TypeVar

from absl import logging
import courier
from ml_metrics._src import base_types
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import func_utils
from ml_metrics._src.utils import iter_utils


_HRTBT_INTERVAL_SECS = 15
_LOGGING_INTERVAL_SEC = 30
_NUM_TOTAL_FAILURES_THRESHOLD = 60
_T = TypeVar('_T')


@functools.lru_cache
def _cached_client(addr: str, call_timeout: int):
  return courier.Client(addr, call_timeout=call_timeout)


@func_utils.lru_cache(settable_kwargs=('max_parallelism', 'iterate_batch_size'))
def cached_worker(
    addr: str,
    *,
    call_timeout: float | None = None,
    max_parallelism: int = 1,
    heartbeat_threshold_secs: float = 180,
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
  def maybe_as_task(cls, task: Task | base_types.Resolvable) -> Task:
    if isinstance(task, base_types.Resolvable):
      return cls.new(task)
    return task

  # The followings are to implement the _FutureLike interfaces.
  def done(self) -> bool:
    """Checks whether the task is done."""
    if (state := self.state) is not None:
      return state.done()
    return False

  def result(self) -> _T:
    if self.state is not None:
      return lazy_fns.maybe_unpickle(self.state.result())

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


@dc.dataclass(kw_only=True, frozen=True)
class GeneratorTask(Task):
  """Courier worker communication for generator.

  Attributes:
    args: The positional arguments later used to pass in the fn.
    kwargs: The named arguments later used to pass in the fn.
    blocking: Whether the wait for the call to complete.
    keep_result: Whether to keep the result.
    server_name: The server address this task is sent to.
    state: the result of the task.
    exception: the exception of the running this task if there is any.
    result: get the result of the task if there is any.
  """


def _as_generator_task(
    task: GeneratorTask | base_types.Resolvable,
) -> GeneratorTask:
  if isinstance(task, base_types.Resolvable):
    return GeneratorTask.new(task)
  return task


MaybeDoneTasks = NamedTuple(
    'DoneNotDoneTask',
    [('done', list[_FutureLike]), ('not_done', list[_FutureLike])],
)


def get_results(
    states: Iterable[_FutureLike[_T]], timeout: float | None = None
) -> list[_T]:
  """Gets the result of the futures, blocking by default."""
  return [
      lazy_fns.maybe_unpickle(task.result())
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
class RemoteObject(Generic[_T], base_types.Resolvable[_T]):
  """Remote object holds remote reference that behaves like a local object."""
  value: lazy_fns.LazyObject[_T]
  worker_addr: str = dc.field(kw_only=True)
  call_timeout: float | None = dc.field(kw_only=True, default=None)

  @classmethod
  def new(
      cls,
      value,
      *,
      worker: str | Worker,
      call_timeout: float | None = None,
  ) -> Self:
    if not isinstance(value, lazy_fns.LazyObject):
      value = lazy_fns.LazyObject.new(value)
    worker_addr = worker.server_name if isinstance(worker, Worker) else worker
    return cls(value, worker_addr=worker_addr, call_timeout=call_timeout)

  def __hash__(self) -> int:
    return hash(self.value)

  def __eq__(self, other: Self) -> bool:
    return self.value == other.value

  def __str__(self) -> str:
    return f'<@{self.worker_addr}:object(id={self.id})>'

  @property
  def id(self) -> int:
    assert (v := self.value) is not None
    return v.id

  @property
  def worker(self) -> Worker:
    return cached_worker(self.worker_addr, call_timeout=self.call_timeout)

  def future_(self) -> futures.Future[bytes]:
    # Remote object will forward any exception as the result.
    return self.worker.call(self.value, return_exception=True)

  def result_(self) -> _T:
    return self.worker.get_result(self.value)

  # TODO: b/356633410 - implements async in process runner.
  async def async_result_(self) -> _T:
    return await self.worker.async_get_result(self.value)

  # Overrides to support pickling when getattr is overridden.
  def __getstate__(self):
    return dict(self.__dict__)

  # Overrides to support pickling when getattr is overridden.
  def __setstate__(self, state):
    self.__dict__.update(state)

  # The following are remote builtin methods.
  def __call__(self, *args, **kwargs) -> RemoteObject:
    """Calling a LazyFn records a lazy result of the call."""
    return RemoteObject.new(
        self.value(*args, **kwargs),
        worker=self.worker_addr,
        call_timeout=self.call_timeout,
    )

  def __getattr__(self, name: str) -> RemoteObject:
    return RemoteObject.new(
        getattr(self.value, name),
        worker=self.worker_addr,
        call_timeout=self.call_timeout,
    )

  def __getitem__(self, key) -> RemoteObject:
    return RemoteObject.new(
        self.value[key], worker=self.worker_addr, call_timeout=self.call_timeout
    )

  # TODO: b/356633410 - Replaces RemoteIterator with RemoteIteratorQueue using
  # courier_server.make_remote_queue.
  def __iter__(self) -> RemoteIterator:
    remote_iterator = self.worker.get_result(
        lazy_fns.trace(iter)(self.value, lazy_result_=True)
    )
    return RemoteIterator(remote_iterator)

  # TODO: b/356633410 - Replaces RemoteIterator with RemoteIteratorQueue using
  # courier_server.make_remote_queue.
  def __aiter__(self) -> RemoteIterator:
    return self.__iter__()


class RemoteIteratorQueue(iter_utils.AsyncIterableQueue[_T]):
  """Remote iterator queue that implements AsyncIteratorQueue interfaces."""
  _queue: RemoteObject[iter_utils.IteratorQueue[_T]]

  def __init__(
      self, q: RemoteObject[iter_utils.IteratorQueue[_T]], *, name: str = ''
  ):
    self._queue = q
    self.name = name

  def get(self):
    logging.debug('chainable: remote queue "%s" get', self.name)
    return self._queue.get().result_()

  async def async_get(self):
    logging.debug('chainable: remote queue "%s" async_get', self.name)
    return await self._queue.get().async_result_()


class RemoteIterator(Iterator[_T]):
  """A local iterator that iterate through remote iterator."""

  iterator: RemoteObject[Iterator[_T]]

  def __init__(self, iterator: RemoteObject[Iterator[_T]]):
    self.iterator = iterator

  def __next__(self) -> _T:
    return self.iterator.worker.get_result(
        lazy_fns.trace(next)(self.iterator.value)
    )

  def __iter__(self):
    return self

  async def __anext__(self) -> _T:
    return await self.iterator.worker.async_get_result(
        lazy_fns.trace(next)(self.iterator.value)
    )

  def __aiter__(self):
    return self


async def async_remote_iter(
    iterator: base_types.MaybeResolvable[Iterable[_T]],
    *,
    worker: str | Worker | None = None,
    buffer_size: int = 0,
    timeout: float | None = None,
    name: str = '',
) -> RemoteIteratorQueue[_T]:
  """Constructs a remote iterator given a maybe lazy iterator."""
  if isinstance(iterator, RemoteObject):
    worker = iterator.worker
    iterator = iterator.value
  elif isinstance(iterator, lazy_fns.LazyObject):
    if not isinstance(worker, Worker):
      worker = cached_worker(worker, call_timeout=timeout)
  else:
    raise TypeError(f'Unsupported {type(iterator)}.')
  # Create a queue at the worker, this returns a remote object.
  lazy_output_q = await worker.async_get_result(
      lazy_fns.trace(iter_utils.IteratorQueue)(
          buffer_size,
          timeout=timeout,
          name=f'{name}@{worker.server_name}',
          lazy_result_=True,
      )
  )
  # Start the remote worker to enqueue the input_iterator.
  _ = worker.call(lazy_output_q.enqueue_from_iterator(iterator))
  # Wrap this queue to behave like a normal queue.
  return RemoteIteratorQueue(lazy_output_q, name=name)


def _is_queue_full(e: Exception) -> bool:
  # Courier worker returns a str reprenestation of the exception.
  return isinstance(e, queue.Full) or 'queue.Full' in getattr(e, 'message', '')


def is_timeout(e: Exception | None) -> bool:
  # absl::status::kDeadlineExceeded
  return getattr(e, 'code', 0) == 4


_InputAndFuture = NamedTuple(
    'InputAndFuture', [('input_', Any), ('future', futures.Future[bytes])]
)


def _normalize_args(args, kwargs):
  """Normalizes the args and kwargs to be picklable."""
  result_args = []
  for arg in args:
    try:
      result_args.append(lazy_fns.pickler.dumps(arg))
    except Exception as e:
      raise ValueError(f'Having issue pickling arg: {arg}, {type(arg)}') from e
  result_kwargs = {}
  for k, v in kwargs.items():
    try:
      result_kwargs[k] = lazy_fns.pickler.dumps(v)
    except Exception as e:
      raise ValueError(f'Having issue pickling {k}: {v}') from e
  return result_args, result_kwargs


StateWithTime = NamedTuple(
    'StateWithTime', [('state', futures.Future[Any]), ('time', float)]
)


def _is_heartbeat_stale(state_with_time: StateWithTime) -> bool:
  """Checks whether the heartbeat is stale."""
  return time.time() - state_with_time.time > _HRTBT_INTERVAL_SECS


# TODO(b/311207032): Adds unit test to cover logic for disconneted worker.
class Worker:
  """Courier client wrapper that works as a chainable worker."""

  server_name: str
  max_parallelism: int
  heartbeat_threshold_secs: float
  iterate_batch_size: int
  _call_timeout: float | None
  _lock: threading.Lock
  _worker_pool: WorkerPool | None
  _client: courier.Client
  _pendings: list[StateWithTime]
  _heartbeat_client: courier.Client
  _heartbeat: StateWithTime
  _last_heartbeat: float

  def __init__(
      self,
      server_name: str | None = '',
      *,
      call_timeout: float | None = 60,
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
    self._refresh_courier_client()
    self._heartbeat = StateWithTime(
        self._heartbeat_client.futures.heartbeat(), time.time()
    )
    self._pendings = [self._heartbeat]
    self._last_heartbeat = 0.0

  def _refresh_courier_client(self):
    assert self.server_name, f'empty server_name: "{self.server_name}"'
    logging.debug(
        'chainable: refreshing courier client at %s', self.server_name
    )
    self._client = courier.Client(
        self.server_name, call_timeout=self.call_timeout
    )
    self._heartbeat_client = courier.Client(
        self.server_name,
        call_timeout=self.heartbeat_threshold_secs,
        wait_for_ready=False,
    )

  @property
  def call_timeout(self) -> float | None:
    return self._call_timeout

  @property
  def worker_pool(self) -> WorkerPool | None:
    return self._worker_pool

  def __str__(self) -> str:
    return (
        f'Worker({self.server_name}, timeout={self.call_timeout},'
        f' max_parallelism={self.max_parallelism},'
        f' from_last_heartbeat={(time.time() - self._last_heartbeat):.2f}s)'
    )

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
    # TODO: b/356633410 - Build a better capacity check than pendings.
    # Limitation: recursive remote calls will cause deadlock. This is why
    # capacity is not enforced at lower level calls, e.g., `call()` and
    # `result()`, and `async_result()`.
    return len(self.pendings) < self.max_parallelism

  def _send_heartbeat(self):
    """Ping the worker to check the heartbeat once."""
    # Only renews the heartbeat one at a time.
    if self._heartbeat.state.done() and _is_heartbeat_stale(self._heartbeat):
      # It is possible the client is stale, e.g., server is started after the
      # client is created. This is only applicable for the first heartbeat.
      if not self._last_heartbeat:
        self._refresh_courier_client()
      heatbeat_future = self._heartbeat_client.futures.heartbeat()
      self._heartbeat = StateWithTime(heatbeat_future, time.time())
      self._pendings.append(self._heartbeat)

  async def async_wait_until_alive(self, deadline_secs: float = 0.0):
    """Checks whether the worker is alive."""
    if self.is_alive:
      return
    ticker = time.time()
    deadline_secs = deadline_secs or self.heartbeat_threshold_secs
    while (delta_time := time.time() - ticker) < deadline_secs:
      if self.is_alive:
        return
    await asyncio.sleep(0.1)
    raise RuntimeError(
        f'Failed to connect to async worker {self.server_name} after'
        f' {delta_time:.2f}s.'
    )

  def wait_until_alive(
      self,
      deadline_secs: float = 0.0,
  ):
    """Waits for the worker to be alive with retries."""
    if self.is_alive:
      return
    ticker = time.time()
    deadline_secs = deadline_secs or self.heartbeat_threshold_secs
    while (delta_time := time.time() - ticker) < deadline_secs:
      if self.is_alive:
        return
      time.sleep(0.1)
    raise RuntimeError(
        f'Failed to connect to worker {self.server_name} after'
        f' {delta_time:.2f}s.'
    )

  def _is_heartbeat_fresh(self) -> bool:
    """Checks whether the heartbeat is stale."""
    still_pendings = []
    for state_and_time in self._pendings:
      if state_and_time.state.done():
        try:
          if not state_and_time.state.exception():
            self._last_heartbeat = max(
                state_and_time.time, self._last_heartbeat
            )
        except futures.CancelledError:
          # The actual exception will be caught on the callsite.
          time.sleep(0)
      else:
        still_pendings.append(state_and_time)
    self._pendings = still_pendings
    return time.time() - self._last_heartbeat < self.heartbeat_threshold_secs

  @property
  def is_alive(self) -> bool:
    """Checks whether the worker is alive."""
    if not self._is_heartbeat_fresh():
      # No last heartbeat recorded, consider it not alive temporarily.
      self._send_heartbeat()
      return False
    return True

  @property
  def pendings(self) -> list[StateWithTime]:
    """Returns the states that are not done."""
    return [p for p in self._pendings if not p.state.done()]

  def call(
      self, *args, courier_method: str = 'maybe_make', **kwargs
  ) -> futures.Future[Any]:
    """Low level courier call ignoring capacity and lock."""
    args, kwargs = _normalize_args(args, kwargs)
    assert self._client is not None
    state = getattr(self._client.futures, courier_method)(*args, **kwargs)
    self._pendings.append(StateWithTime(state, time.time()))
    return state

  def _result_or_exception(self, pickled: bytes) -> Any | Exception:
    result = lazy_fns.pickler.loads(pickled)
    if isinstance(result, Exception):
      raise result
    return result

  def get_result(self, lazy_obj: base_types.Resolvable[_T]) -> _T:
    """Low level blocking courier call to retrieve the result."""
    self.wait_until_alive()
    future = self.call(lazy_obj, return_exception=True)
    while not future.done():
      if not self.is_alive:
        raise RuntimeError(f'Worker disconnected: {self}')
      time.sleep(0)
    try:
      return self._result_or_exception(future.result())
    except Exception as e:  # pylint: disable=broad-exception-caught
      if is_timeout(e):
        if self.is_alive:
          raise TimeoutError(f'Try longer timeout on {self}') from e
        else:
          e.add_note(f'Courier worker {self} died.')
      raise e

  async def async_get_result(self, lazy_obj: base_types.Resolvable[_T]) -> _T:
    """Low level async courier call to retrieve the result."""
    await self.async_wait_until_alive()
    future = self.call(lazy_obj, return_exception=True)
    while not future.done():
      if not self.is_alive:
        raise RuntimeError(f'Async worker disconnected: {self}')
      await asyncio.sleep(0)
    try:
      return self._result_or_exception(future.result())
    except StopIteration as e:
      raise StopAsyncIteration(*e.args) from e
    except Exception as e:  # pylint: disable=broad-exception-caught
      if is_timeout(e):
        if self.is_alive:
          raise TimeoutError(f'Try longer timeout on {self}') from e
        else:
          e.add_note(f'Courier worker {self} died.')
      raise e

  def submit(self, task: Task[_T] | base_types.Resolvable[_T]) -> Task[_T]:
    """Runs tasks sequentially and returns the task."""
    self.wait_until_alive()
    if isinstance(task, base_types.Resolvable):
      task = Task.new(task)
    while not self.has_capacity:
      time.sleep(0)
    state = self.call(
        *task.args, courier_method=task.courier_method, **task.kwargs
    )
    if task.blocking:
      futures.wait([state])
    return task.set(state=state, worker=self)

  def next_batch_from_generator(
      self, batch_size: int = 0
  ) -> futures.Future[Any]:
    batch_size = batch_size or self.iterate_batch_size
    return self.call(
        courier_method='next_batch_from_generator', batch_size=batch_size
    )

  def next_from_generator(self) -> futures.Future[Any]:
    return self.call(courier_method='next_from_generator')

  # TODO: b/356633410 - Deprecate async_iterate in favor of async_iter.
  async def async_iterate(
      self,
      task: GeneratorTask,
      *,
      generator_result_queue: queue.SimpleQueue[transform.AggregateResult],
  ) -> AsyncIterator[Any]:
    """Iterates the generator task."""
    # Artificially insert a pending state to block other tasks.
    generator_state = futures.Future()
    self._pendings.append(StateWithTime(generator_state, time.time()))
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
          'chainable: worker %s generator exhausted after %d batches',
          self.server_name,
          batch_cnt,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception('chainable: exception when iterating task: %s', task)
      raise e
    finally:
      generator_state.cancel()

  async def async_iter(
      self,
      lazy_iterable: lazy_fns.LazyObject[Iterable[_T]],
      *,
      buffer_size: int = 0,
      timeout: float | None = None,
      name: str = '',
  ) -> AsyncIterator[_T]:
    """Async iterates the generator task."""
    batch_cnt = 0
    remote_iterator = await async_remote_iter(
        lazy_iterable,
        worker=self,
        timeout=timeout,
        buffer_size=buffer_size,
        name=name,
    )
    logging.info('chainable: %s remote iterator constructed.', self.server_name)
    async for batch in remote_iterator:
      yield batch
      batch_cnt += 1
      logging.debug(
          'chainable: "%s" async iter yield %d batch of a type %s.',
          self.server_name,
          batch_cnt,
          type(batch),
      )
    logging.info(
        'chainable: remote iterator at %s exhausted after %d batches',
        self.server_name,
        batch_cnt,
    )

  def clear_cache(self):
    return self.call(courier_method='clear_cache')

  def cache_info(self):
    return lazy_fns.pickler.loads(
        self.call(courier_method='cache_info').result()
    )

  def shutdown(self) -> futures.Future[Any]:
    self.state = self._client.futures.shutdown()
    for p in self._pendings:
      p.state.cancel()
    self._pendings = []
    self._last_heartbeat = 0.0
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
      self._workers = [cached_worker(name) for name in names_or_workers]
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
          logging.info('chainable: pool connected %d workers', len(workers))
          return
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('chainable: exception when connecting: %s', type(e))
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
  def num_workers(self):
    return len(self.server_names)

  def shutdown(self, blocking: bool = False):
    """Attemping to shut down workers."""
    while blocking and len(self.workers) < self.num_workers:
      logging.info(
          'chainable: shutting down %d workers, remaining %d workers are not'
          ' connected, retrying.',
          self.num_workers,
          self.num_workers - len(self.workers),
      )
      time.sleep(6)
    logging.info(
        'chainable: shutting down %d workers, remaining %d workers are not'
        ' connected, needs to be manually shutdown.',
        len(self.workers),
        len(self.all_workers) - len(self.workers),
    )
    states = [worker.shutdown() for worker in self.all_workers]
    time.sleep(0.1)
    return states

  def run(self, task: lazy_fns.LazyFn | Task) -> Any:
    """Run lazy object or task within the worker pool."""
    self.wait_until_alive()
    worker = self.next_idle_worker(maybe_acquire=True)
    if worker is None:
      raise ValueError('No worker is available.')
    # Always set blocking to True as run is blocking.
    task = Task.maybe_as_task(task).set(blocking=True)
    result = worker.submit(task).result()
    worker.release()
    return result

  # TODO: b/356633410 - Deprecate iterate.
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
              'chainable: submitting task to worker %s', worker.server_name
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
            if isinstance(exc, TimeoutError) or is_timeout(exc):
              disconnected_tasks.append(task)
            else:
              logging.exception(
                  'chainable: task failed with exception: %s, task: %s',
                  task.exception(),
                  task,
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
            'chainable: %d tasks failed, %d tasks disconnected.',
            len(failed_tasks),
            len(disconnected_tasks),
        )
        tasks.extend(disconnected_tasks)
        total_failures_cnt += len(failed_tasks)
        if total_failures_cnt > num_total_failures_threshold:
          logging.exception(
              'chainable: too many failures: %d > %d, stopping, '
              'last exception:\n %s.',
              total_failures_cnt,
              num_total_failures_threshold,
              failed_tasks[-1].exception(),
          )
          raise failed_tasks[-1].exception()
        else:
          tasks.extend(failed_tasks)
          logging.info(
              'chainable: %d tasks failed, re-trying: %s.',
              len(failed_tasks),
              failed_tasks,
          )
      running_workers = {task.worker for task in running_tasks if task.worker}

      if (ticker := time.time()) - prev_ticker > _LOGGING_INTERVAL_SEC:
        logging.info(
            'chainable: async throughput: %.2f batches/s; progress:'
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
