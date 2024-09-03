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
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import iter_utils


_DEFAULT = 'default'
_T = TypeVar('_T')
pickler = lazy_fns.pickler


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

    def shutdown():
      self._shutdown_requested[_DEFAULT] = True

    def pickled_maybe_make(maybe_lazy):
      result = lazy_fns.maybe_make(maybe_lazy)
      if isinstance(result, lazy_fns.LazyObject):
        result = courier_worker.RemoteObject.new(result, worker=self.address)
      return pickler.dumps(result)

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
