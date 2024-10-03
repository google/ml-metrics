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

import dataclasses
import threading
import time
from typing import Any, Iterable, TypeVar

from absl import logging
import courier
from ml_metrics._src import base_types
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import func_utils
from ml_metrics._src.utils import iter_utils


_T = TypeVar('_T')
pickler = lazy_fns.pickler


@func_utils.lru_cache(settable_kwargs=('timeout_secs',))
def _cached_server(name: str | None = None, *, timeout_secs: float = 10200):
  result = CourierServerWrapper(name, timeout_secs=timeout_secs)
  result.build_server()
  result.start(daemon=True)
  return result


def make_remote_iterator(
    iterator: base_types.MaybeResolvable[Iterable[_T]],
    *,
    server_addr: str,
) -> courier_worker.RemoteIterator[_T]:
  """Constructs a remote iterator given a maybe lazy iterator."""
  iterator = lazy_fns.maybe_make(iterator)
  # Creates a local iterator that can be served as remote iterator.
  return courier_worker.RemoteIterator(
      courier_worker.RemoteObject.new(iter(iterator), worker=server_addr)
  )


def make_remote_object(
    q: base_types.MaybeResolvable[iter_utils.IteratorQueue[_T]],
    *,
    server_addr: str,
) -> courier_worker.RemoteObject[_T]:
  """Constructs a remote queue given a maybe lazy queue."""
  q = lazy_fns.maybe_make(q)
  return courier_worker.RemoteObject.new(q, worker=server_addr)


def make_remote_queue(
    q: base_types.MaybeResolvable[iter_utils.IteratorQueue[_T]],
    *,
    server_addr: str,
    name: str = '',
) -> courier_worker.RemoteIteratorQueue[_T]:
  """Constructs a remote queue given a maybe lazy queue."""
  q = lazy_fns.maybe_make(q)
  assert isinstance(q, iter_utils.IteratorQueue), f'got {type(q)}'
  name = name or q.name
  return courier_worker.RemoteIteratorQueue.from_queue(
      courier_worker.RemoteQueue(
          make_remote_object(q, server_addr=server_addr)
      ),
      name=name,
  )


@dataclasses.dataclass
class CourierServerWrapper:
  """Courier server that runs a chainable."""

  server_name: str | None = None
  port: int | None = None
  prefetch_size: int = 2
  timeout_secs: float = 10200
  _server: courier.Server | None = None
  _thread: threading.Thread | None = None
  _stats: dict[str, float] = dataclasses.field(default_factory=dict)
  _shutdown_requested: bool = False
  _generator: iter_utils.PrefetchedIterator | None = dataclasses.field(
      default=None, init=False
  )

  @property
  def address(self) -> str:
    if self.server_name:
      return self.server_name
    assert self._server is not None, 'Server is not built.'
    return self._server.address

  @property
  def has_started(self) -> bool:
    return self._server is not None and self._server.has_started

  def set_up(self) -> None:
    """Set up (e.g. binding to methods) at server build time."""

    def shutdown():
      self._shutdown_requested = True

    def pickled_maybe_make(maybe_lazy, return_exception: bool = False):
      try:
        result = lazy_fns.maybe_make(maybe_lazy)
      except Exception as e:  # pylint: disable=broad-exception-caught
        lazy_obj = f'{lazy_fns.pickler.loads(maybe_lazy)}'
        if not return_exception:
          raise e
        result = e
        if isinstance(e, (ValueError, TypeError, RuntimeError)):
          logging.exception(
              'chainable: server side exception for %s.', lazy_obj
          )
      if isinstance(result, lazy_fns.LazyObject):
        result = courier_worker.RemoteObject.new(result, worker=self.address)
      try:
        return pickler.dumps(result)
      except TypeError as e:
        lazy_obj = f'{lazy_fns.pickler.loads(maybe_lazy)}'
        logging.exception(
            'chainable: server side pickle error for %s from %s',
            type(result),
            lazy_obj,
        )
        raise e

    def pickled_init_iterator(maybe_lazy):
      result = lazy_fns.maybe_make(maybe_lazy)
      if not isinstance(result, Iterable):
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

    def _next_batch_from_iterator(batch_size: int = 0) -> list[Any]:
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

    def next_batch_from_iterator(batch_size: int | bytes = 0) -> bytes:
      batch_size = lazy_fns.maybe_make(batch_size)
      return pickler.dumps(_next_batch_from_iterator(batch_size))

    def next_from_iterator() -> bytes:
      return pickler.dumps(_next_batch_from_iterator(batch_size=1)[0])

    def heartbeat() -> None:
      self._stats['last_heartbeat'] = time.time()

    assert self._server is not None, 'Server is not built.'
    self._server.Bind('maybe_make', pickled_maybe_make)
    self._server.Bind('init_generator', pickled_init_iterator)
    self._server.Bind('next_from_generator', next_from_iterator)
    self._server.Bind('next_batch_from_generator', next_batch_from_iterator)
    self._server.Bind('heartbeat', heartbeat)
    self._server.Bind('shutdown', shutdown)
    # TODO: b/318463291 - Add unit tests.
    self._server.Bind('clear_cache', transform.clear_cache)
    self._server.Bind('cache_info', pickled_cache_info)

  def build_server(self) -> courier.Server:
    """Build and run a courier server."""
    assert self.server_name != '', f'illegal {self.server_name=}'  # pylint: disable=g-explicit-bool-comparison
    if self._server is not None:
      return self._server
    self._shutdown_requested = False
    self._server = courier.Server(self.server_name, port=self.port)
    self.set_up()
    logging.info('chainable: building server %s', self.address)
    return self._server

  def run_until_shutdown(self):
    """Run until shutdown requested."""
    self.build_server()
    assert self._server is not None, 'Server is not built.'
    if not self._server.has_started:
      self._server.Start()
    self._stats['last_heartbeat'] = time.time()
    while not self._shutdown_requested:
      if time.time() - self._stats['last_heartbeat'] > self.timeout_secs:
        logging.info('Chainables: no ping after %ds.', self.timeout_secs)
        self._shutdown_requested = True
      if self._generator:
        self._generator.prefetch()
      time.sleep(0)
    logging.info(
        'Chainables: Shutdown requested, shutting down server %s',
        self._server.address,
    )
    self._server.Stop()
    self._server = None

  def start(self, daemon: bool = None) -> threading.Thread:
    """Start the server from a different thread."""
    if self.has_started and self._thread is not None:
      return self._thread
    self.build_server()
    server_thread = threading.Thread(
        target=self.run_until_shutdown, daemon=daemon
    )
    server_thread.start()
    self._thread = server_thread
    return server_thread

  def wait_until_alive(self, deadline_secs: float = 120):
    """Wait until the server is alive."""
    courier_worker.cached_worker(self.address).wait_until_alive(
        deadline_secs=deadline_secs
    )
