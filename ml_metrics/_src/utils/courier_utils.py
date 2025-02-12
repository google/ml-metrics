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
"""Courier utils including server and client."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator, Iterable, Iterator
from concurrent import futures
import dataclasses as dc
import gzip
import queue
import threading
import time
from typing import Any, Generic, NamedTuple, Protocol, Self, TypeVar

from absl import logging
import courier
from ml_metrics._src import types
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import func_utils
from ml_metrics._src.utils import iter_utils


_HRTBT_INTERVAL_SECS = 30
_HRTBT_THRESHOLD_SECS = 360
_T = TypeVar('_T')


class FutureLike(Protocol[_T]):

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
class Task(FutureLike[_T]):
  """Lazy function that runs on courier methods.

  Attributes:
    args: The positional arguments later used to pass in the fn.
    kwargs: The named arguments later used to pass in the fn.
    blocking: Whether the wait for the call to complete.
    keep_result: Whether to keep the result.
    worker: The courier worker that runs this task.
    state: The result of the task.
    courier_method: The courier method of the task.
    exception: The exception of the running this task if there is any.
    result: get The result of the task if there is any.
    server_name: The server address this task is sent to.
    is_alive: True if the worker running this task is alive.
  """

  args: tuple[Any, ...] = ()
  kwargs: dict[str, Any] = dc.field(default_factory=dict)
  blocking: bool = False
  worker: CourierClient | None = None
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
  def maybe_as_task(cls, task: Task | types.Resolvable) -> Task:
    if isinstance(task, Task):
      return task
    return cls.new(task)

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
    assert (worker := self.worker) is not None
    return worker.address

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
    state: The result of the task.
    exception: The exception of the running this task if there is any.
    result: Get the result of the task if there is any.
  """


@dc.dataclass(frozen=True)
class RemoteObject(Generic[_T], types.Resolvable[_T]):
  """Remote object holds remote reference that behaves like a local object."""

  value: lazy_fns.LazyObject[_T]
  client_configs: ClientConfig

  @classmethod
  def new(cls, value, worker: str | CourierClient | ClientConfig) -> Self:
    if not isinstance(value, lazy_fns.LazyObject):
      value = lazy_fns.LazyObject.new(value)
    if isinstance(worker, str):
      worker = CourierClient(worker).configs
    elif isinstance(worker, CourierClient):
      worker = worker.configs
    elif not isinstance(worker, ClientConfig):
      raise TypeError(f'Unsupported worker type: {type(worker)}')
    return cls(value, worker)

  def __hash__(self) -> int:
    return hash(self.value)

  def __eq__(self, other: Self) -> bool:
    return self.value == other.value

  def __str__(self) -> str:
    return f'<@{self.client_configs.address}:object(id={self.id})>'

  @property
  def id(self) -> int:
    assert (v := self.value) is not None
    return v.id

  @property
  def worker(self) -> CourierClient:
    return self.client_configs.make()

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
        self.value(*args, **kwargs), worker=self.client_configs
    )

  def __getattr__(self, name: str) -> RemoteObject:
    return RemoteObject.new(
        getattr(self.value, name), worker=self.client_configs
    )

  def __getitem__(self, key) -> RemoteObject:
    return RemoteObject.new(self.value[key], worker=self.client_configs)

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

  @classmethod
  def new(
      cls,
      q: types.MaybeResolvable[iter_utils.IteratorQueue[_T]],
      *,
      server_addr: str | CourierClient | ClientConfig,
      name: str = '',
  ) -> Self:
    q = lazy_fns.maybe_make(q)
    assert isinstance(q, iter_utils.IteratorQueue), f'got {type(q)}'
    name = name or q.name
    return cls(RemoteObject.new(q, worker=server_addr), name=name)

  def get(self):
    logging.debug('chainable: remote queue "%s" get', self.name)
    return self._queue.get().result_()

  async def async_get(self):
    logging.debug('chainable: remote queue "%s" async_get', self.name)
    return await self._queue.get().async_result_()

  def get_batch(self):
    result = self._queue.get_batch().result_()
    logging.debug(
        'chainable: %s',
        f'remote queue "{self.name}" get {len(result)} elems',
    )
    return result

  async def async_get_batch(self):
    result = await self._queue.get_batch().async_result_()
    logging.debug(
        'chainable: %s',
        f'remote queue "{self.name}" async get {len(result)} elems',
    )
    return result


class RemoteIterator(Iterator[_T]):
  """A local iterator that iterate through remote iterator."""

  iterator: RemoteObject[Iterator[_T]]

  def __init__(self, iterator: RemoteObject[Iterator[_T]]):
    self.iterator = iterator

  @classmethod
  def new(
      cls,
      iterator: types.MaybeResolvable[Iterable[_T]],
      *,
      server_addr: str | CourierClient | ClientConfig,
  ) -> Self:
    iterator = lazy_fns.maybe_make(iterator)
    return cls(RemoteObject.new(iter(iterator), worker=server_addr))

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
    iterator: types.MaybeResolvable[Iterable[_T]],
    *,
    worker: str | CourierClient | None = None,
    buffer_size: int = 0,
    timeout: float | None = None,
    name: str = '',
) -> RemoteIteratorQueue[_T]:
  """Constructs a remote iterator given a maybe lazy iterator."""
  if isinstance(iterator, RemoteObject):
    worker = iterator.worker
    iterator = iterator.value
  elif isinstance(iterator, lazy_fns.LazyObject):
    if not isinstance(worker, CourierClient):
      worker = CourierClient(worker, call_timeout=timeout)
  else:
    raise TypeError(f'Unsupported {type(iterator)}.')
  return await worker.async_iter(
      iterator,
      buffer_size=buffer_size,
      timeout=timeout,
      name=name,
  )


def is_timeout(e: Exception | None) -> bool:
  # absl::status::kDeadlineExceeded
  return getattr(e, 'code', 0) == 4


def _maybe_pickle(obj: Any) -> Any:
  # Relying on courier's own pickler for primitives.
  if type(obj) in (str, int, float, bool, type(None)):
    return obj
  try:
    return lazy_fns.pickler.dumps(obj)
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise ValueError(f'Having issue pickling {obj}') from e


def _normalize_args(args, kwargs):
  """Normalizes the args and kwargs to be picklable."""
  result_args = [_maybe_pickle(arg) for arg in args]
  result_kwargs = {k: _maybe_pickle(v) for k, v in kwargs.items()}
  return result_args, result_kwargs


StateWithTime = NamedTuple(
    'StateWithTime', [('state', futures.Future[Any]), ('time', float)]
)


def _is_heartbeat_stale(
    state_with_time: StateWithTime, interval: float
) -> bool:
  """Checks whether the heartbeat is stale."""
  return time.time() - state_with_time.time > interval


@dc.dataclass(frozen=True, kw_only=True, eq=True)
class ClientConfig:
  address: str
  max_parallelism: int
  heartbeat_threshold_secs: float
  iterate_batch_size: int
  call_timeout: float | None

  def make(self) -> CourierClient:
    return CourierClient(**dc.asdict(self))


class CourierClient(metaclass=func_utils.SingletonMeta):
  """Courier client wrapper that works as a chainable worker."""

  address: str
  max_parallelism: int
  heartbeat_threshold_secs: float
  iterate_batch_size: int
  _call_timeout: float
  _lock: threading.Lock
  # "_states_lock" guards all the variables below.
  _states_lock: threading.RLock
  _client: courier.Client
  _pendings: list[StateWithTime]
  _heartbeat_client: courier.Client
  _heartbeat: StateWithTime | None
  _client_heartbeat: float

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
    self.address = address
    self.max_parallelism = max_parallelism
    self.heartbeat_threshold_secs = heartbeat_threshold_secs
    self.iterate_batch_size = iterate_batch_size
    self._call_timeout = call_timeout
    self._lock = threading.Lock()
    self._states_lock = threading.RLock()
    self._refresh_clients()
    self._heartbeat = None
    self._pendings = []
    self._client_heartbeat = 0.0

  @property
  def configs(self) -> ClientConfig:
    return ClientConfig(
        address=self.address,
        max_parallelism=self.max_parallelism,
        heartbeat_threshold_secs=self.heartbeat_threshold_secs,
        call_timeout=self._call_timeout,
        iterate_batch_size=self.iterate_batch_size,
    )

  @property
  def _last_heartbeat(self) -> float:
    return self._client_heartbeat

  @_last_heartbeat.setter
  def _last_heartbeat(self, value: float):
    self._client_heartbeat = max(value, self._client_heartbeat)

  def _refresh_clients(self):
    assert self.address, f'empty address: "{self.address}"'
    logging.debug('chainable: refresh courier client at %s', self.address)
    self._client = courier.Client(self.address, call_timeout=self.call_timeout)
    self._heartbeat_client = courier.Client(
        self.address,
        call_timeout=self.heartbeat_threshold_secs,
    )

  @property
  def call_timeout(self) -> float | None:
    return self._call_timeout

  def __str__(self) -> str:
    return (
        f'{self.__class__.__name__}("{self.address}",'
        f' timeout={self.call_timeout}, max_parallelism={self.max_parallelism},'
        f' from_last_heartbeat={(time.time() - self._last_heartbeat):.2f}s)'
    )

  def __hash__(self):
    return hash(self.address)

  def __eq__(self, other: Any):
    return isinstance(other, CourierClient) and self.configs == other.configs

  @call_timeout.setter
  def call_timeout(self, call_timeout: float):
    with self._states_lock:
      self._call_timeout = call_timeout
      self._client = courier.Client(
          self.address, call_timeout=self.call_timeout
      )

  @property
  def has_capacity(self) -> bool:
    # TODO: b/356633410 - Build a better capacity check than pendings.
    # Limitation: recursive remote calls will cause deadlock. This is why
    # capacity is not enforced at lower level calls, e.g., `call()` and
    # `result()`, and `async_result()`.
    return len(self.pendings) < self.max_parallelism

  def send_heartbeat(
      self, client_address: str, is_alive: bool = True
  ) -> futures.Future[None]:
    """Sends the heartbeat to inform the host at the address is (not) alive."""
    if not client_address:
      raise ValueError('client address is empty')
    if not self.is_alive:
      self._refresh_clients()
    return self._heartbeat_client.futures.heartbeat(client_address, is_alive)

  # TODO: b/376480832 - Deprecate actively checking heartbeat.
  def _check_heartbeat(self, interval: float = _HRTBT_INTERVAL_SECS):
    """Ping the worker to check the heartbeat once."""
    # Only renews the heartbeat one at a time.
    if not self._heartbeat or (
        self._heartbeat.state.done()
        and _is_heartbeat_stale(self._heartbeat, interval)
    ):
      # It is possible the client is stale, e.g., server is started after the
      # client is created. This is only applicable for the first heartbeat.
      self._refresh_clients()
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
        f'Failed to connect to async worker {self.address} after'
        f' {delta_time:.2f}s.'
    )

  def wait_until_alive(
      self,
      deadline_secs: float = 0.0,
      check_capacity: bool = False,
  ):
    """Waits for the worker to be alive with retries."""
    if self.is_alive:
      return
    ticker = time.time()
    deadline_secs = deadline_secs or self.heartbeat_threshold_secs
    while (delta_time := time.time() - ticker) < deadline_secs:
      has_capacity_ = not check_capacity or self.has_capacity
      if self.is_alive and has_capacity_:
        return
      time.sleep(0.1)
    raise RuntimeError(
        f'Failed to connect to worker {self.address} after {delta_time:.2f}s.'
    )

  def _is_heartbeat_fresh(self) -> bool:
    """Checks whether the heartbeat is stale."""
    still_pendings = []
    for state_and_time in self._pendings:
      if state_and_time.state.done():
        try:
          if not state_and_time.state.exception():
            self._last_heartbeat = state_and_time.time
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
    with self._states_lock:
      if self._is_heartbeat_fresh():
        return True
      # No last heartbeat recorded, consider it not alive temporarily.
      self._check_heartbeat()
      return False

  @property
  def pendings(self) -> list[StateWithTime]:
    """Returns the states that are not done."""
    with self._states_lock:
      return [p for p in self._pendings if not p.state.done()]

  def call(
      self, *args, courier_method: str = 'maybe_make', **kwargs
  ) -> futures.Future[Any]:
    """Low level courier call ignoring capacity and lock."""
    args, kwargs = _normalize_args(args, kwargs)
    assert self._client is not None
    with self._states_lock:
      state = getattr(self._client.futures, courier_method)(*args, **kwargs)
      self._pendings.append(StateWithTime(state, time.time()))
    return state

  def _result_or_exception(self, pickled: bytes) -> Any | Exception:
    result = lazy_fns.pickler.loads(gzip.decompress(pickled))
    if isinstance(result, Exception):
      raise result
    if isinstance(result, lazy_fns.LazyObject):
      return RemoteObject.new(result, worker=self)
    return result

  def get_result(self, lazy_obj: types.Resolvable[_T]) -> _T:
    """Low level blocking courier call to retrieve the result."""
    self.wait_until_alive()
    future = self.call(lazy_obj, return_exception=True, compress=True)
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

  async def async_get_result(self, lazy_obj: types.Resolvable[_T]) -> _T:
    """Low level async courier call to retrieve the result."""
    await self.async_wait_until_alive()
    future = self.call(lazy_obj, return_exception=True, compress=True)
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

  def submit(self, task: Task[_T] | types.Resolvable[_T]) -> Task[_T]:
    """Runs tasks sequentially and returns the task."""
    self.wait_until_alive()
    if not isinstance(task, Task):
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
      generator_result_queue: queue.SimpleQueue[Any],
  ) -> AsyncIterator[Any]:
    """Iterates the generator task."""
    # Artificially insert a pending state to block other tasks.
    with self._states_lock:
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
            generator_result_queue.put(returned := stop_iteration.value)
            output_batch = output_batch[:-1]
            logging.info(
                'chainable: %s',
                f'worker "{self.address}" generator exhausted after'
                f' {batch_cnt} batches with return type {type(returned)}',
            )
          elif isinstance(exc := output_batch[-1], Exception):
            if exc != ValueError('generator already executing'):
              raise output_batch[-1]
        for elem in output_batch:
          yield elem
          batch_cnt += 1
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
  ) -> RemoteIteratorQueue[_T]:
    """Async iterates the generator task."""
    # Create a queue at the worker, this returns a remote object.
    lazy_output_q = await self.async_get_result(
        lazy_fns.trace(iter_utils.IteratorQueue)(
            buffer_size,
            timeout=timeout,
            name=f'{name}@{self.address}',
            lazy_result_=True,
        )
    )
    # Start the remote worker to enqueue the input_iterator.
    _ = self.call(
        lazy_output_q.enqueue_from_iterator(lazy_iterable),
        return_exception=True,
        ignore_result=True,
    )
    # Wrap this queue to behave like a normal queue.
    return RemoteIteratorQueue(lazy_output_q, name=f'{name}@{self.address}')

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
    self._client_heartbeat = 0.0
    return self.state
