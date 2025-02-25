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

from collections.abc import Callable
from concurrent import futures
import signal
import threading
import time
from typing import Any, Iterable, TypeVar
import weakref

from absl import logging
import courier
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import courier_utils
from ml_metrics._src.utils import func_utils
from ml_metrics._src.utils import iter_utils


_T = TypeVar('_T')
_WeakRefDict = weakref.WeakKeyDictionary[_T, weakref.ref[_T]]

pickler = lazy_fns.pickler
_HRTBT_INTERVAL_SECS = 60
_THREAD_POOL = futures.ThreadPoolExecutor()


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


class _CourierServerSingleton(func_utils.SingletonMeta):
  """Singleton metaclass for CourierServer."""

  _instances: _WeakRefDict[CourierServer] = weakref.WeakKeyDictionary()

  @property
  def all_instances(self) -> list[CourierServer]:
    return super().all_instances


class CourierServer(metaclass=_CourierServerSingleton):
  """Courier server that runs a chainable."""

  server_name: str | None
  port: int | None
  auto_shutdown_secs: float
  _states_lock: threading.Lock
  _server: courier.Server | None
  _thread: threading.Thread | None
  _last_heartbeat: float
  _shutdown_lock: threading.Condition
  _shutdown_requested: bool
  _heartbeats: HeartbeatRegistry
  _shutdown_callback: Callable[..., None] | None
  _clients: list[courier_utils.CourierClient]
  _thread_pool: futures.ThreadPoolExecutor

  def __init__(
      self,
      server_name: str | None = None,
      *,
      port: int | None = None,
      auto_shutdown_secs: float = 10200,
      clients: Iterable[str] = (),
  ):
    self.server_name = server_name
    self.port = port
    self._server = None
    self._thread = None
    self._last_heartbeat = 0.0
    self._shutdown_lock = threading.Condition()
    self._shutdown_requested = False
    self._states_lock = threading.Lock()
    self._heartbeats = HeartbeatRegistry()
    self._shutdown_callback = None
    self._tx_stats_lock = threading.Lock()
    self._tx_stats = (time.time(), 0, 0)
    self.auto_shutdown_secs = auto_shutdown_secs
    self.build_server()
    self.set_shutdown_callback(self.notify_shutdown)
    self._clients = [courier_utils.CourierClient(addr) for addr in clients]
    self._thread_pool = _THREAD_POOL
    try:
      signal.signal(signal.SIGINT, self.notify_shutdown)
      signal.signal(signal.SIGTERM, self.notify_shutdown)
    except ValueError:
      logging.warning(
          'chainable: cannot register signal handler for %s, try constructing'
          ' Server from the main thread.',
          self,
      )

  def __str__(self):
    addr = self.server_name or ''
    if self._server is not None:
      addr += f'@{self._server.address}'
    return f'{self.__class__.__name__}("{addr}")'

  def __hash__(self) -> int:
    return hash(self.address)

  def __eq__(self, other: Any) -> bool:
    return (
        isinstance(other, CourierServer)
        and self.address == other.address
        and self.auto_shutdown_secs == other.auto_shutdown_secs
    )

  def __del__(self):
    self.notify_shutdown()

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

  def _notify_alive(self, is_alive: bool = True):
    for client in self._clients:
      client.send_heartbeat(self.address, is_alive=is_alive)
      logging.info(
          'chainable: notify alive=%s from "%s" to "%s"',
          is_alive,
          self.address,
          client.address,
      )

  @property
  def has_started(self) -> bool:
    return (
        self._server is not None
        and self._server.has_started
        and self._thread is not None
    )

  def notify_shutdown(self, signum=None, frame=None):
    """Notify the server to shutdown, also served as a signal handler.

    Args:
      signum: not used, but is required for this to be a signal handler.
      frame: not used, but is required for this to be a signal handler.
    """
    del signum, frame
    with self._shutdown_lock:
      self._shutdown_requested = True
      self._shutdown_lock.notify_all()

  def set_up(self) -> None:
    """Set up (e.g. binding to methods) at server build time."""

    def pickled_maybe_make(
        maybe_lazy,
        return_exception: bool = False,
        compress: bool = False,
        ignore_result: bool = False,
    ):
      self._last_heartbeat = time.time()
      try:
        # ignore_result still honors other arguements compress and
        # return_exception where optionally compress the result (None) and
        # returns the exception if threadpool.submit fails.
        if ignore_result:
          self._thread_pool.submit(lazy_fns.maybe_make, maybe_lazy)
          result = None
        else:
          result = lazy_fns.maybe_make(maybe_lazy)
      except Exception as e:  # pylint: disable=broad-exception-caught
        lazy_obj = f'{lazy_fns.pickler.loads(maybe_lazy)}'
        if not return_exception:
          raise e
        result = e
        if isinstance(e, (ValueError, TypeError, RuntimeError)):
          logging.exception('chainable: maybe_make exception for %s.', lazy_obj)
      try:
        result = pickler.dumps(result, compress=compress)
        with self._tx_stats_lock:
          ticker, bytes_sent, num_txs = self._tx_stats
          self._tx_stats = (ticker, bytes_sent + len(result), num_txs + 1)
        return result
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

  def build_server(self) -> courier.Server:
    """Build and run a courier server."""
    assert self.server_name != '', f'illegal {self.server_name=}'  # pylint: disable=g-explicit-bool-comparison
    if self._server is None:
      self._shutdown_requested = False
      self._server = courier.Server(self.server_name, port=self.port)
      self.set_up()
    return self._server

  def set_shutdown_callback(self, callback: Callable[..., None]):
    self._shutdown_callback = callback

  def _shutdown_server(self):
    self._notify_alive(is_alive=False)
    with self._states_lock:
      if self.has_started:
        assert self._server is not None, 'Server is not built.'
        if self._shutdown_callback is not None:
          self._shutdown_callback()
        logging.info('chainable: Shutting down server %s', self)
        self._server.Stop()
        self._server = None

  def run_until_shutdown(self):
    """Run until shutdown requested."""
    courier_server = self.build_server()
    if not courier_server.has_started:
      courier_server.Start()
    self._last_heartbeat = time.time()
    with self._shutdown_lock:
      while not self._shutdown_requested:
        if time.time() - self._last_heartbeat > self.auto_shutdown_secs:
          logging.info('chainable: no ping after %ds.', self.auto_shutdown_secs)
          self._shutdown_requested = True
          break
        self._shutdown_lock.wait(_HRTBT_INTERVAL_SECS)
        with self._tx_stats_lock:
          ticker, bytes_sent, num_txs = self._tx_stats
          delta_time = time.time() - ticker
          logging.info(
              'chainable: %s',
              f'"{self.address}" tx stats: {bytes_sent} bytes,'
              f' {num_txs} requests in {delta_time:.2f} seconds,'
              f' ({bytes_sent / delta_time:.2f} bytes/sec,'
              f' {num_txs / delta_time:.2f} requests/sec).',
          )
          self._tx_stats = (time.time(), 0, 0)
    self._shutdown_server()

  def start(self, *, daemon: bool = True) -> threading.Thread:
    """Start the server from a different thread."""
    del daemon  # Depreate daemon.
    with self._states_lock:
      courier_server = self.build_server()
      if not courier_server.has_started:
        courier_server.Start()
      if not self._thread:
        self._thread = threading.Thread(
            target=self.run_until_shutdown, daemon=True
        )
        self._thread.start()
    assert self._thread is not None
    self._notify_alive()
    return self._thread

  def stop(self) -> threading.Thread:
    """Stop the server."""
    self.notify_shutdown()
    assert self._thread is not None
    return self._thread


def client_heartbeat(address: str) -> float:
  """Check the last heartbeat of from all servers created by CourierServer."""
  heartbeats = (
      obj.client_heartbeat(address) for obj in CourierServer.all_instances
  )
  return max(heartbeats, default=0.0)


def shutdown_all(*, block: bool = True):
  """Shutdown all instances created by CourierServer."""
  servers = [
      (s, s.address) for s in CourierServer.all_instances if s.has_started
  ]
  logging.info('chainable: shutting down %d servers.', len(servers))
  for server, _ in servers:
    server.stop()
  if block:
    for server, address in servers:
      server.stop().join()
      logging.info('chainable: server %s stopped.', address)


class PrefetchedCourierServer(CourierServer):
  """Courier server that can prefetch an iterator."""

  prefetch_size: int
  _enqueue_thread: threading.Thread | None
  _generator: iter_utils.IteratorQueue | None

  def __init__(
      self,
      server_name: str | None = None,
      *,
      port: int | None = None,
      # TODO: b/372935688 - Rename this to auto_shutdown_secs.
      timeout_secs: float = 10200,
      prefetch_size: int = 2,
      clients: Iterable[str] = (),
  ):
    super().__init__(
        server_name,
        port=port,
        auto_shutdown_secs=timeout_secs,
        clients=clients,
    )
    self.prefetch_size = prefetch_size
    self._enqueue_thread = None
    self._generator = None

  def __hash__(self) -> int:
    return super().__hash__()

  def __eq__(self, other: Any) -> bool:
    return (
        super().__eq__(other)
        and isinstance(other, PrefetchedCourierServer)
        and self.prefetch_size == other.prefetch_size
    )

  def set_up(self) -> None:
    """Set up (e.g. binding to methods) at server build time."""

    def pickled_init_iterator(maybe_lazy):
      self._last_heartbeat = time.time()
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
      self._last_heartbeat = time.time()
      if self._generator is None:
        raise RuntimeError(
            'Generator is not set, the worker might be killed previously, the'
            ' task normally will be restarted. This could be caused by worker'
            ' killed by the Borglet. Search for "chainable:.+retries" in the'
            ' log to see how persistent this is.'
        )
      result = []
      try:
        result = self._generator.get_batch(batch_size, block=True)
        with self._tx_stats_lock:
          ticker, bytes_sent, num_txs = self._tx_stats
          self._tx_stats = (ticker, bytes_sent + len(result), num_txs + 1)
      except Exception as e:  # pylint: disable=broad-exception-caught
        if not iter_utils.is_stop_iteration(e):
          logging.exception('chainable: exception when flushing generator.')
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


# TODO: b/372935688 - Deprecate CourierServerWrapper.
CourierServerWrapper = PrefetchedCourierServer
