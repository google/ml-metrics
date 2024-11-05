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
from __future__ import annotations

import threading
import time
from typing import Any, Iterable, TypeVar

from absl import logging
import courier
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import func_utils
from ml_metrics._src.utils import iter_utils


_T = TypeVar('_T')
pickler = lazy_fns.pickler
_HRTBT_INTERVAL_SECS = 30


class HeartbeatRegistry:
  """A singleton class to track the clients of a server."""

  _clients: dict[str, float]

  def __init__(self):
    self._clients = {}

  def get(self, address: str):
    """Get the last heartbeat of a client."""
    return self._clients.get(address, None)

  def update(self, address: str, time_: float):
    """Update the heartbeat of any existing client."""
    if address in self._clients:
      self._clients[address] = max(self._clients.get(address, 0), time_)
    raise ValueError(f'address {address} is not registered.')

  def register(self, address: str, time_: float):
    """Register a new client."""
    self._clients[address] = time_

  def unregister(self, address: str):
    """Unregister a client."""
    self._clients.pop(address, None)


class Server:
  """Courier server that runs a chainable."""

  server_name: str | None
  port: int | None
  _states_lock: threading.Lock
  _server: courier.Server | None
  _thread: threading.Thread | None
  _last_heartbeat: float
  _generator: iter_utils.IteratorQueue | None
  _enqueue_thread: threading.Thread | None
  _shutdown_lock: threading.Condition
  _shutdown_requested: bool
  _heartbeats: HeartbeatRegistry

  def __init__(
      self,
      server_name: str | None = None,
      *,
      port: int | None = None,
  ):
    self.server_name = server_name
    self.port = port
    self._server = None
    self._thread = None
    self._last_heartbeat = 0.0
    self._shutdown_lock = threading.Condition()
    self._shutdown_requested = False
    self._generator = None
    self._enqueue_thread = None
    self._states_lock = threading.Lock()
    self._heartbeats = HeartbeatRegistry()
    self.build_server()

  def __str__(self):
    addr = self.server_name or ''
    if self._server is not None:
      addr += f'@{self._server.address}'
    return f'CourierServer("{addr}")'

  def __hash__(self) -> int:
    return hash(self.address)

  def __eq__(self, other: Server) -> bool:
    return self.address == other.address

  def client_heartbeat(self, client_addrs: str) -> float:
    return self._heartbeats.get(client_addrs) or 0.0

  @property
  def address(self) -> str:
    """The server has to be built before accessing the address."""
    if server_name := self.server_name:
      return server_name
    if self._server is None:
      raise RuntimeError('Server is not built, the address is not available.')
    return self._server.address

  @property
  def has_started(self) -> bool:
    return self._server is not None and self._server.has_started

  def notify_shutdown(self):
    with self._shutdown_lock:
      self._shutdown_requested = True
      self._shutdown_lock.notify_all()

  def set_up(self) -> None:
    """Set up (e.g. binding to methods) at server build time."""

    def pickled_maybe_make(maybe_lazy, return_exception: bool = False):
      try:
        result = lazy_fns.maybe_make(maybe_lazy)
      except Exception as e:  # pylint: disable=broad-exception-caught
        lazy_obj = f'{lazy_fns.pickler.loads(maybe_lazy)}'
        if not return_exception:
          raise e
        result = e
        if isinstance(e, (ValueError, TypeError, RuntimeError)):
          logging.exception('chainable: maybe_make exception for %s.', lazy_obj)
      try:
        return pickler.dumps(result)
      except TypeError as e:
        lazy_obj = f'{lazy_fns.pickler.loads(maybe_lazy)}'
        logging.exception(
            'chainable: maybe_make pickle error for %s from %s',
            type(result),
            lazy_obj,
        )
        raise e

    def pickled_cache_info():
      return pickler.dumps(lazy_fns.cache_info())

    def heartbeat(sender_addr: str = '', is_alive: bool = True) -> None:
      self._last_heartbeat = time.time()
      if not sender_addr:
        return
      with self._states_lock:
        if is_alive:
          self._heartbeats.register(sender_addr, self._last_heartbeat)
        else:
          self._heartbeats.unregister(sender_addr)

    assert self._server is not None, 'Server is not built.'
    self._server.Bind('maybe_make', pickled_maybe_make)
    self._server.Bind('heartbeat', heartbeat)
    self._server.Bind('shutdown', self.notify_shutdown)
    self._server.Bind('clear_cache', transform.clear_cache)
    self._server.Bind('cache_info', pickled_cache_info)

  def build_server(self, start_server: bool = False) -> courier.Server:
    """Build and run a courier server."""
    assert self.server_name != '', f'illegal {self.server_name=}'  # pylint: disable=g-explicit-bool-comparison
    if self._server is None:
      self._shutdown_requested = False
      self._server = courier.Server(self.server_name, port=self.port)
      self.set_up()
      logging.info('chainable: constructed server %s', self)
    if start_server and not self.has_started:
      self._server.Start()
    return self._server

  def shutdown_server(self):
    if self.has_started:
      logging.info('chainable: Shutting down for server %s', self)
      assert self._server is not None, 'Server is not built.'
      self._server.Stop()
      self._server = None


# TODO: b/372935688 - Rename this to PrefetchingServer.
class CourierServerWrapper(Server):
  """Courier server that can prefetch an iterator."""

  prefetch_size: int
  timeout_secs: float

  def __init__(
      self,
      server_name: str | None = None,
      *,
      port: int | None = None,
      prefetch_size: int = 2,
      timeout_secs: float = 10200,
  ):
    super().__init__(server_name, port=port)
    self.prefetch_size = prefetch_size
    self.timeout_secs = timeout_secs

  def __hash__(self) -> int:
    return super().__hash__()

  def __eq__(self, other: CourierServerWrapper) -> bool:
    if not isinstance(other, CourierServerWrapper):
      return False
    return (
        self.address == other.address
        and self.prefetch_size == other.prefetch_size
        and self.timeout_secs == other.timeout_secs
    )

  def set_up(self) -> None:
    """Set up (e.g. binding to methods) at server build time."""

    def pickled_init_iterator(maybe_lazy):
      result = lazy_fns.maybe_make(maybe_lazy)
      if not isinstance(result, Iterable):
        raise TypeError(
            f'The {result} is not a generator, but a {type(result)}.'
        )
      if self._generator is not None and not self._generator.exhausted:
        logging.warning(
            'chainable: A new generator is initialized while the previous one'
            ' is not exhausted.'
        )
        self._generator.stop_enqueue()
        if self._enqueue_thread:
          self._enqueue_thread.join()
      self._generator = iter_utils.IteratorQueue(
          self.prefetch_size,
          ignore_error=True,
          name=f'prefetch_queue@{self.address}',
      )
      self._enqueue_thread = threading.Thread(
          target=self._generator.enqueue_from_iterator,
          args=(result,),
          daemon=True,
      )
      self._enqueue_thread.start()
      with self._shutdown_lock:
        self._shutdown_lock.notify_all()

    def _next_batch_from_iterator(batch_size: int = 0) -> list[Any]:
      assert self._generator is not None, (
          'Generator is not set, the worker might crashed unexpectedly'
          ' previously.'
      )
      try:
        result = self._generator.flush(batch_size, block=True)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('chainable: exception when flushing generator.')
        raise e
      # The sequence of the result will always end with an exception.
      # Any non-StopIteration means the generator crashed.
      if not self._generator:
        if self._generator.exception:
          result.append(self._generator.exception)
        else:
          result.append(StopIteration(*self._generator.returned))
      return result

    def next_batch_from_iterator(batch_size: int | bytes = 0) -> bytes:
      batch_size = lazy_fns.maybe_make(batch_size)
      return pickler.dumps(_next_batch_from_iterator(batch_size))

    def next_from_iterator() -> bytes:
      return pickler.dumps(_next_batch_from_iterator(batch_size=1)[0])

    assert self._server is not None, 'Server is not built.'
    super().set_up()
    self._server.Bind('init_generator', pickled_init_iterator)
    self._server.Bind('next_from_generator', next_from_iterator)
    self._server.Bind('next_batch_from_generator', next_batch_from_iterator)

  def run_until_shutdown(self):
    """Run until shutdown requested."""
    self.build_server(start_server=True)
    self._last_heartbeat = time.time()
    with self._shutdown_lock:
      while not self._shutdown_requested:
        if time.time() - self._last_heartbeat > self.timeout_secs:
          logging.info('chainable: no ping after %ds.', self.timeout_secs)
          self._shutdown_requested = True
          break
        self._shutdown_lock.wait(_HRTBT_INTERVAL_SECS)
    self.shutdown_server()

  def start(self, *, daemon: bool = None) -> threading.Thread:
    """Start the server from a different thread."""
    with self._states_lock:
      if self.has_started:
        assert self._thread is not None, 'Threading is having issue locking.'
        return self._thread
      self.build_server()
      server_thread = threading.Thread(
          target=self.run_until_shutdown, daemon=daemon
      )
      server_thread.start()
      self._thread = server_thread
      return server_thread

  def stop(self) -> threading.Thread | None:
    """Stop the server."""
    self.notify_shutdown()
    return self._thread


@func_utils.lru_cache(settable_kwargs=('timeout_secs',))
def _cached_server(
    name: str | None = None, *, timeout_secs: float = 10200
):
  result = CourierServerWrapper(name, timeout_secs=timeout_secs)
  result.build_server()
  result.start(daemon=True)
  return result
