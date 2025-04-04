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

_HRTBT_INTERVAL_SECS = 60
_THREAD_POOL = futures.ThreadPoolExecutor()

# Thread-safe registry of workers that are alive notified either by servers or
# clients. The registry should not be mutated by any other means. Assign to
# _worker_registry directly will raise TypeError.
_worker_registry = courier_utils.WorkerRegistry()


def worker_heartbeat(address: str) -> float:
  """Check the last heartbeat of from all servers created by CourierServer."""
  return _worker_registry.get(address)


def worker_registry() -> courier_utils.WorkerRegistry:
  """Get the worker registry."""
  return _worker_registry


def _register_worker(address: str, time_: float):
  """Register a new worker."""
  _worker_registry.register(address, time_)


def _unregister_worker(address: str):
  """Unregister a worker."""
  _worker_registry.unregister(address)


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
    self._server: courier.Server | None = None
    self._thread: threading.Thread | None = None
    self._last_heartbeat = 0.0
    self._shutdown_lock = threading.Condition()
    self._shutdown_requested = False
    self._states_lock = threading.Lock()
    self._shutdown_callback: Callable[..., None] | None = None
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
      signal.signal(signal.SIGABRT, self.notify_shutdown)
    except ValueError:
      logging.warning(
          'chainable: %s',
          f'cannot register signal handler for {self}, try constructing Server'
          ' from the main thread.',
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
          'chainable: %s',
          f'notify {is_alive=} from "{self.address}" to "{client.address}"',
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
    logging.warning('chainable: %s', f'notify_shutdown {signum=}')
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
        return_immediately: bool = False,
        return_none: bool = False,
    ):
      """Maybe make a lazy object.

      Args:
        maybe_lazy: The lazy object to be made.
        return_exception: Whether to return the server-side exception when
          exception is raised during the call.
        compress: Whether to return the compressed result, this is useful to
          lower the traffic size.
        return_immediately: Whether to ignore the result and return immediately.
        return_none: Whether to return None instead of the actual result. This
          is useful to avoid expensive items to be pickled and sent back.

      Returns:
        The result of the maybe_make call.

      Raises:
        Exception: If return_exception is False and the call raises an
          exception.
      """
      self._last_heartbeat = time.time()
      try:
        maybe_lazy = lazy_fns.maybe_unpickle(maybe_lazy)
        # return_immediately still honors other arguements compress and
        # return_exception where optionally compress the result (None) and
        # returns the exception if threadpool.submit fails.
        if return_immediately:
          self._thread_pool.submit(lazy_fns.maybe_make, maybe_lazy)
          result = None
        elif return_none:
          # Skip return the actual result but still wait for finish.
          _ = lazy_fns.maybe_make(maybe_lazy)
          result = None
        else:
          result = lazy_fns.maybe_make(maybe_lazy)
      except Exception as e:  # pylint: disable=broad-exception-caught
        if self._shutdown_requested:
          e = TimeoutError('Shutdown requested, the worker is shutting down.')
        if not return_exception:
          raise e
        result = e
        if isinstance(e, (ValueError, TypeError, RuntimeError)):
          logging.exception(
              'chainable: %s', f'maybe_make exception for {maybe_lazy}.'
          )
      try:
        result = lazy_fns.pickler.dumps(result, compress=compress)
        with self._tx_stats_lock:
          ticker, bytes_sent, num_txs = self._tx_stats
          self._tx_stats = (ticker, bytes_sent + len(result), num_txs + 1)
        return result
      except TypeError as e:
        logging.exception(
            'chainable: %s',
            f'maybe_make pickle error for {type(result)} from {maybe_lazy}',
        )
        raise e

    def pickled_cache_info():
      return lazy_fns.pickler.dumps(lazy_fns.cache_info())

    def heartbeat(sender_addr: str = '', is_alive: bool = True) -> None:
      self._last_heartbeat = time.time()
      if not sender_addr:
        return
      if is_alive:
        _register_worker(sender_addr, self._last_heartbeat)
        logging.info(
            'chainable: %s', f'notified alive=True from "{sender_addr}"'
        )
      else:
        _unregister_worker(sender_addr)
        logging.info(
            'chainable: %s', f'notified alive=False from "{sender_addr}"'
        )

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
        logging.info('chainable: %s', f'Shutting down server {self}')
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
          logging.info(
              'chainable: %s', f'no ping after {self.auto_shutdown_secs}s.'
          )
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


def shutdown_all(*, block: bool = True):
  """Shutdown all instances created by CourierServer."""
  servers = [
      (s, s.address) for s in CourierServer.all_instances if s.has_started
  ]
  logging.info('chainable: %s', f'shutting down {len(servers)} servers.')
  for server, _ in servers:
    server.stop()
  if block:
    for server, address in servers:
      server.stop().join()
      logging.info('chainable: %s', f'server {address} stopped.')


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
      ignore_error: bool = False,
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
    self._ignore_error = ignore_error

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
      if self._shutdown_requested:
        return TimeoutError('Shutdown requested, cannot take new generator.')

      start_time = time.time()
      maybe_lazy = lazy_fns.maybe_unpickle(maybe_lazy)
      logging.info('chainable: %s', f'Initializing generator for {maybe_lazy}')
      result = lazy_fns.maybe_make(maybe_lazy)
      if not isinstance(result, Iterable):
        raise TypeError(
            f'The {result} is not a generator, but a {type(result)}.'
        )
      if self._generator is not None and not self._generator.exhausted:
        logging.warning(
            'chainable: %s',
            'A new generator is initialized while the previous is unexhausted.',
        )
        self._generator.maybe_stop(
            TimeoutError(
                'A new generator is initialized while the previous is'
                ' unexhausted.'
            )
        )
        if self._enqueue_thread and self._enqueue_thread.is_alive():
          self._enqueue_thread.join()
      self._generator = iter_utils.IteratorQueue(
          self.prefetch_size,
          ignore_error=self._ignore_error,
          name=f'prefetch_queue@{self.address}',
      )
      self._enqueue_thread = threading.Thread(
          target=self._generator.enqueue_from_iterator,
          args=(result,),
          daemon=True,
      )
      self._enqueue_thread.start()
      delta_time = time.time() - start_time
      logging.info(
          'chainable: %s',
          f'Generator loaded in {delta_time:.2f}s for {maybe_lazy}',
      )
      with self._shutdown_lock:
        self._shutdown_lock.notify_all()

    def _next_batch_from_iterator(batch_size: int = 0) -> list[Any]:
      if self._generator is None:
        e = TimeoutError(
            'Generator is not set, the worker might be killed previously, the'
            ' task normally will be restarted. This could be caused by worker'
            ' killed by the Borglet. Search for "chainable:.+retries" in the'
            ' log to see how persistent this is.'
        )
        logging.exception('chainable: %s', f'{e}')
        return [e]

      result = []
      try:
        result = self._generator.get_batch(batch_size, block=True)
        with self._tx_stats_lock:
          ticker, bytes_sent, num_txs = self._tx_stats
          self._tx_stats = (ticker, bytes_sent + len(result), num_txs + 1)
      except Exception:  # pylint: disable=broad-exception-caught
        # The sequence of the result will always end with an exception.
        # Any non-StopIteration means the generator crashed. The exception
        # is then appended to the result list and returned in one batch below.
        pass

      if not self._generator:
        if (e := self._generator.exception) is not None:
          if self._shutdown_requested:
            e = TimeoutError('Shutdown requested, cannot get next batch.')
          logging.exception(
              'chainable: %s',
              f'Exception during next batch call: {type(e)}: {e}',
          )
          result.append(e)
        else:
          result.append(StopIteration(*self._generator.returned))
      return result

    def next_batch_from_iterator(batch_size: int | bytes = 0) -> bytes:
      self._last_heartbeat = time.time()
      batch_size = lazy_fns.maybe_make(batch_size)
      return lazy_fns.pickler.dumps(_next_batch_from_iterator(batch_size))

    def next_from_iterator() -> bytes:
      self._last_heartbeat = time.time()
      return lazy_fns.pickler.dumps(_next_batch_from_iterator(batch_size=1)[0])

    assert self._server is not None, 'Server is not built.'
    super().set_up()
    self._server.Bind('init_generator', pickled_init_iterator)
    self._server.Bind('next_from_generator', next_from_iterator)
    self._server.Bind('next_batch_from_generator', next_batch_from_iterator)


# TODO: b/372935688 - Deprecate CourierServerWrapper.
CourierServerWrapper = PrefetchedCourierServer
