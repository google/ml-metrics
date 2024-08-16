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
"""Courier server that can run a chainable."""

import collections
from concurrent import futures
import copy
import dataclasses
import functools
import inspect
import queue
import random
import threading
import time
from typing import Any, Generic, Iterable, Iterator, Self, TypeVar

from absl import logging
import courier
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import iter_utils
import more_itertools as mit


_DEFAULT = 'default'
_T = TypeVar('_T')
pickler = lazy_fns.pickler


@dataclasses.dataclass(frozen=True)
class RemoteObject(Generic[_T]):
  """Remote object holds remote reference that behaves like a local object."""

  worker_address: str
  lazy_object: lazy_fns.LazyObject[_T] | lazy_fns.LazyFn[_T]
  _state: futures.Future[Any] | None = None

  @functools.cached_property
  def _client(self) -> courier.Client:
    client = courier.Client(self.worker_address)
    assert client.heartbeat() is None
    return client

  def deref_(self) -> futures.Future[Any]:
    return self._client.futures.maybe_make(pickler.dumps(self.lazy_object))

  def value_(self) -> Any:
    return pickler.loads(self.deref_().result())

  def set_(self, **kwargs):
    return dataclasses.replace(
        self, lazy_object=self.lazy_object.set_(**kwargs)
    )

  def __hash__(self):
    return hash(self.lazy_object.id)

  def __eq__(self, other):
    return self.lazy_object.id == other.lazy_object.id

  def __call__(self, *args, **kwargs) -> Self:
    """Calling a LazyFn records a lazy result of the call."""
    return dataclasses.replace(
        self, lazy_object=self.lazy_object(*args, **kwargs)
    )

  def __getattr__(self, name) -> Self:
    return dataclasses.replace(
        self, lazy_object=getattr(self.lazy_object, name)
    )

  def __getitem__(self, key) -> Self:
    return dataclasses.replace(self, lazy_object=self.lazy_object[key])

  # Overrides to support pickling when getattr is overridden.
  def __getstate__(self):
    return dict(self.__dict__)

  # Overrides to support pickling when getattr is overridden.
  def __setstate__(self, state):
    self.__dict__.update(state)


def _is_queue_full(e: Exception) -> bool:
  # Courier worker returns a str reprenestation of the exception.
  return 'queue.Full' in getattr(e, 'message', '')


def _is_courier_timeout(exc: Exception | None) -> bool:
  # absl::status::kDeadlineExceeded
  return getattr(exc, 'code', 0) == 4


_InputAndFuture = collections.namedtuple(
    '_InputAndFuture', ['input_', 'future']
)


@dataclasses.dataclass(slots=True)
class RemoteQueues:
  """Combine multiple remote queues into one as iterators when en/dequeueing."""

  remote_queues: set[RemoteObject[queue.Queue[Any]]] = dataclasses.field(
      default_factory=set
  )
  timeout_secs: int | None = None

  def __post_init__(self):
    self.remote_queues = set(self.remote_queues)

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
            result = pickler.loads(state.result())
            del states[remote_queue]
            if isinstance(result, StopIteration):
              reamining_queues.remove(remote_queue)
            else:
              yield result
              ticker = time.time()
          except Exception as e:  # pylint: disable=broad-exception-caught
            # Courier worker returns a str reprenestation of the exception.
            if 'queue.Empty' in getattr(e, 'message', ''):
              if time.time() - ticker > self.timeout_secs:
                raise TimeoutError(
                    f'Dequeue timeout after {self.timeout_secs}s.'
                ) from e
              continue
            else:
              raise e
      time.sleep(0)

  def enqueue(self, values: Iterable[Any]) -> Iterator[Any]:
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
            assert pickler.loads(state.result()) is None
            ticker = time.time()
            preferred_queues.add(remote_queue)
            del enqueueing[remote_queue]
            if not iter_utils.is_stop_iteration(input_):
              yield input_
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


@dataclasses.dataclass
class CourierServerWrapper:
  """Courier server that runs a chainable."""

  server_name: str | None = None
  port: int | None = None
  prefetch_size: int = 2
  timeout_secs: int = 10200
  _server: courier.Server | None = None
  _stats: dict[str, float] = dataclasses.field(default_factory=dict)
  _shutdown_requested: dict[str, bool] = dataclasses.field(
      default_factory=lambda: {_DEFAULT: False}, init=False
  )
  _generator: iter_utils.PrefetchedIterator | None = dataclasses.field(
      default=None, init=False
  )

  @property
  def address(self) -> str:
    return self._server.address if self._server else ''

  def set_up(self) -> None:
    """Set up (e.g. binding to methods) at server build time."""
    # Verify the registered makeables.
    assert lazy_fns.makeables[lazy_fns.LazyFn] is not None

    def shutdown():
      self._shutdown_requested[_DEFAULT] = True

    def pickled_maybe_make(maybe_lazy):
      result = lazy_fns.maybe_make(maybe_lazy)
      if isinstance(result, lazy_fns.LazyObject):
        result = RemoteObject(self.address, result)
      return pickler.dumps(result)

    def pickled_init_generator(maybe_lazy):
      result = lazy_fns.maybe_make(maybe_lazy)
      if not inspect.isgenerator(result):
        raise TypeError(
            f'The {result} is not a generator, but a {type(result)}.'
        )
      if self._generator is not None and not self._generator.exhausted:
        logging.warning(
            'Chainables: A new generator is initialized while the previous one'
            ' is not exhausted.'
        )
      self._generator = iter_utils.PrefetchedIterator(
          result,
          prefetch_size=self.prefetch_size,
      )

    def pickled_cache_info():
      return pickler.dumps(lazy_fns.cache_info())

    def _next_batch_from_generator(batch_size: int = 0) -> list[Any]:
      assert self._generator is not None, (
          'Generator is not set, the worker might crashed unexpectedly'
          ' previously.'
      )
      result = self._generator.flush_prefetched(batch_size)
      # The sequence of the result will always end with an exception.
      # Any non-StopIteration means the generator crashed.
      if not self._generator.data_size:
        if self._generator.exceptions:
          result.append(self._generator.exceptions[-1])
        elif self._generator.exhausted:
          result.append(StopIteration(self._generator.returned))
      return result

    def next_batch_from_generator(batch_size: int | bytes = 0) -> bytes:
      batch_size = lazy_fns.maybe_make(batch_size)
      return pickler.dumps(_next_batch_from_generator(batch_size))

    def next_from_generator() -> bytes:
      return pickler.dumps(_next_batch_from_generator(batch_size=1)[0])

    def heartbeat() -> None:
      self._stats['last_heartbeat'] = time.time()

    assert self._server is not None, 'Server is not built.'
    self._server.Bind('maybe_make', pickled_maybe_make)
    self._server.Bind('init_generator', pickled_init_generator)
    self._server.Bind('next_from_generator', next_from_generator)
    self._server.Bind('next_batch_from_generator', next_batch_from_generator)
    self._server.Bind('heartbeat', heartbeat)
    self._server.Bind('shutdown', shutdown)
    # TODO: b/318463291 - Add unit tests.
    self._server.Bind('clear_cache', transform.clear_cache)
    self._server.Bind('cache_info', pickled_cache_info)

  def build_server(self) -> courier.Server:
    """Build and run a courier server."""
    self._server = courier.Server(self.server_name, port=self.port)
    self.set_up()
    return self._server

  def run_until_shutdown(self):
    """Run until shutdown requested."""
    if self._server is None:
      self.build_server()
    assert self._server is not None, 'Server is not built.'
    if not self._server.has_started:
      self._server.Start()
    self._stats['last_heartbeat'] = time.time()
    while not self._shutdown_requested[_DEFAULT]:
      if time.time() - self._stats['last_heartbeat'] > self.timeout_secs:
        logging.info('Chainables: no ping after %ds.', self.timeout_secs)
        self._shutdown_requested[_DEFAULT] = True
      if self._generator:
        self._generator.prefetch()
      time.sleep(0)
    logging.info('Chainables: Shutdown requested, shutting down server.')
    self._server.Stop()

  def start(self, daemon: bool = None) -> threading.Thread:
    """Start the server from a different thread."""
    server_thread = threading.Thread(
        target=self.run_until_shutdown, daemon=daemon
    )
    server_thread.start()
    return server_thread
