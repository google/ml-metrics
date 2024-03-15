"""Utils for Chainable runners."""

import dataclasses
import functools
from typing import Any

import cloudpickle as pickle
from ml_metrics._src.chainables import lazy_fns


# TODO: b/318463291 - support heterogeneous (de)serializations methods.
@dataclasses.dataclass
class Picklers:
  """Picklers that can be registered."""

  default = pickle

  def register(self, pickler):
    if hasattr(pickler, 'dumps') and hasattr(pickler, 'loads'):
      self.default = pickler
    else:
      raise TypeError(
          f'Pickler {pickler} of type {type(pickler)} has to have `loads` and'
          ' `dumps` methods.'
      )


picklers = Picklers()


@functools.lru_cache(maxsize=4096)
def _maybe_make_cached(payload: Any):
  """Maybe deserialize a payload and call make when applicable."""
  if isinstance(payload, bytes):
    payload = picklers.default.loads(payload)
  result = lazy_fns.maybe_make(payload)
  return result


# TODO: b/318463291 - Add unit test when cached is True.
def maybe_make(payload: Any, cached: bool = False):
  """Maybe deserialize a payload and call make when applicable."""
  if cached:
    return _maybe_make_cached(payload)
  if isinstance(payload, bytes):
    payload = picklers.default.loads(payload)
  return lazy_fns.maybe_make(payload)


# TODO: b/318463291 - Add unit tests.
def clear_cache():
  """Clear the cache for maybe_make."""
  _maybe_make_cached.cache_clear()


# TODO: b/318463291 - Add unit tests.
def cache_info():
  """Returns the cache info for maybe_make."""
  _maybe_make_cached.cache_info()
