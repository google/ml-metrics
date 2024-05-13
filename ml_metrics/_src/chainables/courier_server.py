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
  _generator: transform.PrefetchableIterator | None = dataclasses.field(
      default=None, init=False
  )

  def build_server(self):
    """Build and run a courier server."""
    # Verify the registered makeables.
    assert lazy_fns.makeables[lazy_fns.LazyFn] is not None

    def shutdown():
      self._shutdown_requested[_DEFAULT] = True

    def pickled_maybe_make(maybe_lazy):
      result = lazy_fns.maybe_make(maybe_lazy)
      if inspect.isgenerator(result):
        self._generator = transform.PrefetchableIterator(
            result, prefetch_size=self.prefetch_size
        )
        return pickler.dumps(repr(result))
      return pickler.dumps(result)

    def next_batch_from_generator():
      assert self._generator is not None, (
          'Generator is not set, the worker might crashed unexpectedly'
          ' previously.'
      )
      result = [next(self._generator) for _ in range(self._generator.data_size)]
      if not result and self._generator.exhausted:
        result = lazy_fns.STOP_ITERATION
      return pickler.dumps(result)

    # TODO: b/318463291 - Considers deprecating in favor of
    # `next_batch_from_generator`.
    def next_from_generator():
      try:
        result = next(self._generator)
      except StopIteration:
        result = lazy_fns.STOP_ITERATION
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('Chainables: Exception while iterating: %s', e)
        result = lazy_fns.STOP_ITERATION
      return pickler.dumps(result)

    server = courier.Server(self.server_name, port=self.port)
    server.Bind('maybe_make', pickled_maybe_make)
    server.Bind('next_from_generator', next_from_generator)
    server.Bind('next_batch_from_generator', next_batch_from_generator)
    server.Bind('shutdown', shutdown)
    # TODO: b/318463291 - Add unit tests.
    server.Bind('clear_cache', lazy_fns.clear_cache)
    server.Bind('cache_info', lazy_fns.cache_info)
    self._server = server
    return self._server

  def run_until_shutdown(self):
    """Run until shutdown requested."""
    assert self._server is not None, 'Server is not built.'
    if not self._server.has_started:
      self._server.Start()
    while not self._shutdown_requested[_DEFAULT]:
      if self._generator:
        self._generator.prefetch()
      time.sleep(0.01)
    logging.info('Chainables: Shutdown requested, shutting down server.')
    self._server.Stop()


def run_courier_server(name=None, port=None, prefetch_size: int = 128):
  # TODO: b/318463291 - Preloaded task to start running prefetching even before
  # master started.
  server_wrapper = CourierServerWrapper(
      server_name=name,
      port=port,
      prefetch_size=prefetch_size,
  )
  server_wrapper.build_server()
  server_wrapper.run_until_shutdown()
