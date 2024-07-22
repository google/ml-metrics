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
import inspect
import threading
import time

from absl import logging
import courier
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform

_DEFAULT = 'default'
pickler = lazy_fns.picklers.default


@dataclasses.dataclass
class CourierServerWrapper:
  """Courier server that runs a chainable."""

  server_name: str | None = None
  port: int | None = None
  prefetch_size: int = 2
  _server: courier.Server | None = None
  _shutdown_requested: dict[str, bool] = dataclasses.field(
      default_factory=lambda: {_DEFAULT: False}, init=False
  )
  _generator: transform.PrefetchedIterator | None = dataclasses.field(
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
      return pickler.dumps(result)

    def pickled_init_generator(maybe_lazy):
      result = lazy_fns.maybe_make(maybe_lazy)
      if not inspect.isgenerator(result):
        raise TypeError(
            f'The {result} is not a generator, but a {type(result)}.'
        )
      self._generator = transform.PrefetchedIterator(
          result,
          prefetch_size=self.prefetch_size,
      )

    def pickled_cache_info():
      return pickler.dumps(lazy_fns.cache_info())

    def _next_batch_from_generator(batch_size: int = 0):
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

    def next_batch_from_generator(batch_size: int | bytes = 0):
      batch_size = lazy_fns.maybe_make(batch_size)
      return pickler.dumps(_next_batch_from_generator(batch_size))

    def next_from_generator():
      return pickler.dumps(_next_batch_from_generator(batch_size=1)[0])

    def heartbeat() -> None:
      pass

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
    while not self._shutdown_requested[_DEFAULT]:
      if self._generator:
        self._generator.prefetch()
      time.sleep(0)
    logging.info('Chainables: Shutdown requested, shutting down server.')
    self._server.Stop()

  def start(self, daemon: bool = None):
    """Start the server from a different thread."""
    server_thread = threading.Thread(
        target=self.run_until_shutdown, daemon=daemon
    )
    server_thread.start()
    return server_thread
