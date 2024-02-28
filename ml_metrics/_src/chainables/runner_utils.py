"""Utils for Chainable runners."""

from collections.abc import Iterable
import dataclasses
import functools
import itertools
from typing import Any, Callable

import cloudpickle as pickle
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform as transform_lib


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


# TODO: b/318463291 - Add unit test when ached is True.
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


def _pickled(cached_args=None):
  """This (de)serializes and instantiate function inputs and outputs.

  This wraps in-process function to handle serialized inputs and serialize the
  function outputs. The inputs are either depickled when the are bytes, and/or
  instantiated if tthey are registered in lazy_fns.makeables.

  Args:
    cached_args: this will dicate which named args will be cached.

  Returns:
    The decorator handling (de)serialization of the inputs and outputs.
  """
  cached_args = cached_args or ()

  def decorator(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
      args = [maybe_make(v) for v in args]
      kwargs = {
          k: maybe_make(v, cached=(k in cached_args)) for k, v in kwargs.items()
      }
      result = fn(*args, **kwargs)
      return picklers.default.dumps(result)

    return wrapped

  return decorator


@_pickled(cached_args=('transform',))
def run_update_state(
    transform: transform_lib.StackedTreeAggregateFn,
    input_iterator: Iterable[Any],
    initial_state=None,
    num_steps: int | None = None,
    record_num_steps: Callable[[int], None] | None = None,
    record_state: Callable[[int, Any], None] | None = None,
    record_busy: Callable[[bool], None] | None = None,
):
  """Run update_state of a transform for `num_steps` and returns the state."""
  state = initial_state or transform.create_state()
  if record_num_steps:
    record_num_steps(-1)
  if record_state:
    record_state(0, state)
  if record_busy:
    record_busy(True)
  for i, batch in enumerate(itertools.islice(input_iterator, num_steps)):
    if i == 0 and record_num_steps:
      record_num_steps(0)
    state = transform.update_state(state, batch)
    if record_num_steps:
      record_num_steps(i + 1)
    if record_state:
      record_state(i, state)
  if record_busy:
    record_busy(False)
  return state


@_pickled(cached_args=('transform',))
def run_merge_states(
    transform: transform_lib.StackedTreeAggregateFn,
    states: list[Any],
):
  return transform.merge_states(states)


@_pickled(cached_args=('transform',))
def run_get_result(
    transform: transform_lib.StackedTreeAggregateFn,
    state: Any,
):
  return transform.get_result(state)
