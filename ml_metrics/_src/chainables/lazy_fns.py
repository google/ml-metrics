# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LazyFns and the functions they can make."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Callable, Hashable, Mapping, Sequence
import dataclasses
import functools
import importlib
import inspect
import json
from typing import Any, Generic, TypeVar

import cloudpickle as pickle

ValueT = TypeVar('ValueT')
Fn = Callable[..., ValueT]

STOP_ITERATION = 'StopIteration_'


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


class _Makers(collections.UserDict):
  """Maker registry."""

  def register(self, type_: type[Any], maker: Callable[..., Any]):
    self.data[repr(type_)] = maker

  def __getitem__(self, type_: type[Any]) -> Callable[..., Any] | None:
    return self.data.get(repr(type_), None)


makeables = _Makers()


def maybe_make(maybe_lazy: Any) -> Any:
  if isinstance(maybe_lazy, bytes):
    maybe_lazy = picklers.default.loads(maybe_lazy)
  # User defined maker as an escape path for custom lazy instances.
  if maker := makeables[type(maybe_lazy)]:
    return maker(maybe_lazy)
  return maybe_lazy


def _as_awaitable(fn: Callable[..., Any], *args, **kwargs):
  if inspect.iscoroutinefunction(fn):
    return fn(*args, **kwargs)
  else:
    return asyncio.to_thread(fn, *args, **kwargs)


def async_iterate_fn(fn):
  """Wraps a callable to apply it on each item in an iterable.

  Note that, we assume only the positional arguements are consuming the
  iterables. All the kwargs are directly passed through.

  Args:
    fn: the function to consume one item in the iteratable.

  Returns:
    A function that consumes the iterable.
  """

  @functools.wraps(fn)
  async def wrapped_fun(*inputs, **kwargs) -> tuple[Any, ...]:
    aws = [
        _as_awaitable(fn, *row_inputs, **kwargs) for row_inputs in zip(*inputs)
    ]
    tasks = [asyncio.create_task(aw) for aw in aws]
    outputs_future = asyncio.gather(*tasks)
    await asyncio.wait_for(outputs_future, timeout=None)
    outputs = outputs_future.result()
    # Only transpose when mutliple items returned.
    if outputs and isinstance(outputs[0], tuple):
      return tuple(zip(*outputs))
    else:
      return outputs

  return wrapped_fun


def iterate_fn(fn):
  """Wraps a callable to apply it on each item in an iterable.

  Note that, we assume only the positional arguements are consuming the
  iterables. All the kwargs are directly passed through.

  Args:
    fn: the function to consume one item in the iteratable.

  Returns:
    A function that consumes the iterable.
  """

  @functools.wraps(fn)
  def wrapped_fun(*inputs, **kwargs) -> tuple[ValueT, ...] | ValueT:
    outputs = [fn(*row_inputs, **kwargs) for row_inputs in zip(*inputs)]
    # Only transpose when mutliple items returned.
    if outputs and isinstance(outputs[0], tuple):
      return tuple(zip(*outputs))
    else:
      return outputs

  return wrapped_fun


def normalize_kwargs(kwargs: Mapping[str, Hashable]):
  return tuple(kwargs.items())


@dataclasses.dataclass(kw_only=True, frozen=True)
class Args:
  """Positional and keyword arguements.

  Attributes:
    args: The positional arguements later used to pass in the fn.
    kwargs: The named arguements later used to pass in the fn.
  """

  args: tuple[Hashable | LazyFn, ...] = ()
  kwargs: tuple[tuple[str, Hashable | LazyFn], ...] = ()


def trace_args(*args, **kwargs) -> Args:
  """Traces positional and keyword arguements."""
  return Args(args=args, kwargs=normalize_kwargs(kwargs))


@dataclasses.dataclass(kw_only=True, frozen=True)
class FnConfig:
  """A readable config that instantiates an in-memory LazyFn.

  The config is a serialized config for the LazyFn. There are two directions
  these are used: a) this is a readable version of the in-memory LazyFn;
  b) this can convert a JSON deserializable config to a LazyFn. E.g.,
  `FnConfig(...).make()` makes a in-memory LazyFn instance.

  Attr:
    fn: The function to be called.
    module: The module where the function is located. If not specified, default
      to the current module.
    args: Positional arguments to be passed to the function.
    kwargs: Keyword arguments to be passed to the function.
  """

  fn: str
  module: str = ''
  args: list[Any] = dataclasses.field(default_factory=list)
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def from_json_str(cls, json_str: str):
    return cls(**json.loads(json_str))

  def make(self):
    if self.module:
      actual_fn = getattr(importlib.import_module(self.module), self.fn)
    else:
      actual_fn = eval(self.fn)  # pylint: disable=eval-used
    return LazyFn.new(actual_fn, args=self.args, kwargs=self.kwargs)


@dataclasses.dataclass(kw_only=True, frozen=True)
class LazyFn(Generic[ValueT], Callable[..., 'LazyFn']):
  """A lazy function that has all the information to be called later.

  Attributes:
    fn: The function to be called.
    args: The positional arguements later used to pass in the fn.
    kwargs: The named arguements later used to pass in the fn.
    cache_result: If True, cache the result of the make LazyFn.
  """

  fn: Callable[..., ValueT] | LazyFn | None = None
  args: tuple[Hashable | LazyFn, ...] = ()
  kwargs: tuple[tuple[str, Hashable | LazyFn], ...] = ()
  cache_result: bool = False

  @classmethod
  def new(
      cls,
      fn: Callable[..., ValueT] | None = None,
      *,
      args: Sequence[Hashable] = (),
      kwargs: Mapping[str, Hashable] | None = None,
      cache_result: bool = False,
  ) -> LazyFn[ValueT]:
    """Normalizes the arguements before constructing a LazyFn."""
    kwargs = (kwargs or {}).items()
    return cls(
        fn=fn, args=tuple(args), kwargs=tuple(kwargs), cache_result=cache_result
    )

  def __call__(self, *args, **kwargs) -> LazyFn[LazyFn]:
    """Calling a LazyFn records a lazy result of the call.

    Note, this does not call the actual function. E.g.,
    ```
    def foo(x):
      return lambda y: x + y
    lazy_foo = LazyFn.new(fn=foo)
    ```
    lazy_foo(1)(2) is the lazy version of foo(1)(2) and
    call(lazy_foo(1)(2)) == foo(1)(2).

    Args:
      *args: positional args.
      **kwargs: keyword args.

    Returns:
      A new LazyFn that traces the call of the current LazyFn.
    """
    return self.__class__(
        fn=_make,
        args=(self,) + args,
        kwargs=normalize_kwargs(kwargs),
    )

  def __getattr__(self, name):
    if name.startswith('__') and name.endswith('__'):
      raise AttributeError
    return self.__class__(
        fn=getattr,
        args=(self, name),
    )

  # Overrides to support pickling when getattr is overridden.
  def __getstate__(self):
    return dict(self.__dict__)

  # Overrides to support pickling when getattr is overridden.
  def __setstate__(self, state):
    self.__dict__.update(state)


@functools.lru_cache(maxsize=256)
def _cached_make(fn: LazyFn | bytes) -> Any:
  """Instantiate a lazy fn to a actual fn when applicable."""
  if isinstance(fn, bytes):
    fn = picklers.default.loads(fn)
  if not fn.fn:
    return None
  args = tuple(maybe_make(arg) for arg in fn.args)
  kwargs = {k: maybe_make(v) for k, v in fn.kwargs}
  if fn.fn is _make:
    assert callable(args[0])
    result = args[0](*args[1:], **kwargs)
  elif callable(fn.fn):
    result = fn.fn(*args, **kwargs)
  else:
    raise TypeError(f'fn is not callable from {fn}.')
  return maybe_make(result)


def _make(fn: LazyFn) -> Any:
  """Instantiate a lazy fn to a actual fn when applicable."""
  if not fn.fn:
    return None
  if fn.cache_result:
    # In case the fn is not hash-able, we use the default pickler to pickle it
    # to bytes first then cache it.
    try:
      return _cached_make(fn)
    except TypeError:
      try:
        return _cached_make(picklers.default.dumps(fn))
      except TypeError as e:
        raise TypeError(f'fn is not picklable: {fn}.') from e

  args = tuple(maybe_make(arg) for arg in fn.args)
  kwargs = {k: maybe_make(v) for k, v in fn.kwargs}
  if fn.fn is _make:
    assert callable(args[0])
    result = args[0](*args[1:], **kwargs)
  elif callable(fn.fn):
    result = fn.fn(*args, **kwargs)
  else:
    raise TypeError(f'fn is not callable from {fn}.')
  return maybe_make(result)


makeables.register(LazyFn, _make)


def clear_cache():
  """Clear the cache for maybe_make."""
  _cached_make.cache_clear()


def cache_info():
  """Returns the cache info for maybe_make."""
  return _cached_make.cache_info()


def trace(
    fn: Callable[..., ValueT], use_cache: bool = False
) -> Callable[..., LazyFn[ValueT]]:
  """Traces a callable to record the function and its arguements.

  A lazy function is the lazy counterpart of the actual function. We can convert
  a lazy function using `lazy`: fn -> lazy(fn) -> lazy_fn. Calling a lazy
  counterpart
  of the function doesn't call the actual function, but record the arguements
  used to call the function later. To call the actual function, uses.
  `lazy_fn.call()`. E.g.,
  ```
  lazy_len = lazy(len)()
  # Then:
  lazy_len.call([1,2,3]) == len([1,2,3])
  ```
  This works recursively if the return of the function is also callable, one
  just needs to call the lazy_function one extra time: e.g.,
  ```
  class Foo:
    a: int
    def __call__(x): return a + x
  # Then:
  lazy_foo = lazy(Foo)(a=1)
  lazy_foo(3).call() == Foo(a=1)(3)
  ```
  The arguement to the call() function is arbitarily bindable; e.g.,
  ```
  lazy_foo = lazy(Foo)(a=1)
  lazy_foo().call(3) == lazy_foo(3).call().
  ```
  The arguments to the lazy_fn can also be a lazy_fn. E.g.,
  ```
  def get_a(): return 'a'
  lazy_a, lazy_Foo = lazy(get_a), lazy(Foo)
  lazy_Foo(a=lazy_a).call('b') == Foo('a')('b') == 'ab'
  ```

  Args:
    fn: The fn to be called lazily.
    use_cache: If True, uses cache for the result of the make LazyFn.

  Returns:
    A function that records the fn and its arguements to be called later.
  """

  def wrapped(*args, **kwargs):
    return LazyFn.new(fn=fn, args=args, kwargs=kwargs, cache_result=use_cache)

  return wrapped
