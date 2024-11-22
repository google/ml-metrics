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
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
import dataclasses as dc
import functools
import importlib
import inspect
import itertools as it
import json
import operator
from typing import Any, Self, TypeVar
import uuid

from absl import logging
import cloudpickle as pickle
from ml_metrics._src import base_types
from ml_metrics._src.utils import func_utils

_KeyT = TypeVar('_KeyT')
_T = TypeVar('_T')
Fn = Callable[..., _T]
_LAZY_OBJECT_CACHE_SIZE = 1024


class LazyObjectMissingError(KeyError):
  """Raised when the local object is missing."""


def _maybe_lru_cache(maxsize: int):
  """Decorator to add LRU cache for LazyObject inputs.

  This only caches the result when `cache_result` in the LazyObject is True.

  Args:
    maxsize: the maximum size of the LRU Cache.

  Returns:
    A decorator that can lru cache function with LazyObject inputs.
  """

  def decorator(fn: Callable[..., Any]):
    lazy_obj_cache = func_utils.LruCache(maxsize=maxsize)

    def wrapped_fn(x: LazyObject[_T]) -> _T:
      if x.cache_result:
        try:
          result = lazy_obj_cache[x]
          logging.debug('chainable: cache hit for type %s.', type(result))
          return result
        except KeyError as e:
          if isinstance(x, LazyFn):
            logging.info('chainable: cache miss for type %s', type(x))
            result = fn(x)
            lazy_obj_cache[x] = result
            return result
          else:
            raise LazyObjectMissingError(
                f'{x} is missing, if this is remote, the worker'
                ' might have been restarted.'
            ) from e
      else:
        return fn(x)

    wrapped_fn.cache_info = lazy_obj_cache.cache_info
    wrapped_fn.cache_clear = lazy_obj_cache.cache_clear
    wrapped_fn.cache_insert = lazy_obj_cache.cache_insert
    return wrapped_fn

  return decorator


def _shortened_uuid(b: bytes, length: int) -> int:
  if len(b) == length:
    return int.from_bytes(b, 'big') << (length * 8)

  new_len = len(b) // 2
  b = int.from_bytes(b[:new_len], 'big') ^ int.from_bytes(b[new_len:], 'big')
  b = b.to_bytes(new_len, 'big')
  return _shortened_uuid(b, length=length)


@dc.dataclass(slots=True)
class IncrementId(Iterator[int]):
  """An ID that increments everytime. Used to track local object."""

  id_len: dc.InitVar[int]
  _inc_iter: Iterator[int] = dc.field(default_factory=it.count, init=False)
  _max_id: int = dc.field(init=False, default=0)
  _half_id_len: int = dc.field(
      default=0,
      init=False,
  )
  _base: int = dc.field(default=0, init=False)

  def __post_init__(self, id_len: int):
    self._half_id_len, residual = divmod(id_len, 2)
    if residual:
      raise ValueError(f'ID length has to be divisible by 2, got {id_len}.')
    self._max_id = (1 << self._half_id_len * 8) - 1
    self._base = _shortened_uuid(uuid.uuid4().bytes, self._half_id_len)

  def __next__(self) -> int:
    next_id = next(self._inc_iter)
    # Reset the id to mimic fixed length int.
    if next_id > self._max_id:
      self._inc_iter = it.count()
      next_id = next(self._inc_iter)
    return self._base + next_id

  def __iter__(self):
    return self


_increment_id = IncrementId(id_len=8)


# TODO: b/318463291 - support heterogeneous (de)serializations methods.
@dc.dataclass
class _Pickler:
  """Pickler that can be registered at run time.."""

  default = pickle

  def register(self, pickler_, /):
    if hasattr(pickler_, 'dumps') and hasattr(pickler_, 'loads'):
      self.default = pickler_
    else:
      raise TypeError(
          f'Pickler {pickler_} of type {type(pickler_)} has to have `loads` and'
          ' `dumps` methods.'
      )

  def dumps(self, value: Any) -> bytes:
    return self.default.dumps(value)

  def loads(self, value: bytes) -> Any:
    return self.default.loads(value)


pickler = _Pickler()


def maybe_unpickle(value: Any) -> Any:
  if isinstance(value, bytes):
    return pickler.loads(value)
  return value


class _Makers(collections.UserDict):
  """Maker registry."""

  def register(self, type_: type[Any], maker: Callable[..., Any]):
    self.data[repr(type_)] = maker

  def __getitem__(self, type_: type[Any]) -> Callable[..., Any] | None:
    return self.data.get(repr(type_), None)


makeables = _Makers()


def _maybe_make(
    maybe_lazy: base_types.MaybeResolvable[_T],
) -> base_types.MaybeResolvable[_T]:
  if isinstance(maybe_lazy, base_types.Resolvable):
    return maybe_lazy.result_()
  if maker := makeables[type(maybe_lazy)]:
    return maker(maybe_lazy)
  return maybe_lazy


def maybe_make(
    maybe_lazy: base_types.MaybeResolvable[_T] | bytes,
) -> base_types.MaybeResolvable[_T]:
  """Dereference a lazy object or lazy function when applicable."""
  maybe_lazy = maybe_unpickle(maybe_lazy)
  return _maybe_make(maybe_lazy)


def _as_awaitable(fn: Callable[..., Any], *args, **kwargs):
  if inspect.iscoroutinefunction(fn):
    return fn(*args, **kwargs)
  else:
    return asyncio.to_thread(fn, *args, **kwargs)


def async_iterate_fn(fn):
  """Wraps a callable to apply it on each item in an iterable.

  Note that, we assume only the positional arguments are consuming the
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
    # Only transpose when multiple items returned.
    if outputs and isinstance(outputs[0], tuple):
      return tuple(zip(*outputs))
    else:
      return outputs

  return wrapped_fun


def iterate_fn(fn) -> Callable[..., tuple[_T, ...] | _T]:
  """Wraps a callable to apply it on each item in an iterable.

  Note that, we assume only the positional arguments are consuming the
  iterables. All the kwargs are directly passed through.

  Args:
    fn: the function to consume one item in the iteratable.

  Returns:
    A function that consumes the iterable.
  """

  @functools.wraps(fn)
  def wrapped_fun(*inputs, **kwargs) -> tuple[_T, ...] | _T:
    outputs = [fn(*row_inputs, **kwargs) for row_inputs in zip(*inputs)]
    # Only transpose when fn returns multiple items (exact tuple).
    if outputs and type(outputs[0]) is tuple:  # pylint: disable=unidiomatic-typecheck
      return tuple(zip(*outputs))
    return outputs

  return wrapped_fun


def normalize_kwargs(kwargs: Mapping[str, Hashable]):
  return tuple(kwargs.items())


@dc.dataclass(kw_only=True, frozen=True)
class FnConfig:
  """A readable config that instantiates an in-memory LazyFn.

  The config is a serialized config for the LazyFn, which can be used in two
  directions:
    a) this is a readable version of the in-memory LazyFn;
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
  args: list[Any] = dc.field(default_factory=list)
  kwargs: dict[str, Any] = dc.field(default_factory=dict)

  @classmethod
  def from_json_str(cls, json_str: str):
    return cls(**json.loads(json_str))

  def make(self):
    if self.module:
      actual_fn = getattr(importlib.import_module(self.module), self.fn)
    else:
      actual_fn = eval(self.fn)  # pylint: disable=eval-used
    return LazyFn.new(actual_fn, args=self.args, kwargs=self.kwargs)


@dc.dataclass(kw_only=True, frozen=True)
class LazyObject(base_types.Resolvable[_T]):
  """A remote object that can be pickled.

  Attributes:
    value: The function to be called.
    is_lazy: If True, the instance value is not stored.
    id: The id of the LazyFn that persists across processes.
    cache_result: If True, cache the result of the make LazyFn.
    lazy_result: If True, return a LazyObject instead of the actual result.
  """

  value: _T | None = None
  _cache_result: bool = False
  _lazy_result: bool = False
  _id: int = dc.field(default_factory=lambda: next(_increment_id), init=False)

  @classmethod
  def new(
      cls,
      value: _T,
      *,
      cache_result: bool = True,
      lazy_result: bool = False,
  ):
    """Creates and LazyObject, optionally with the value stored locally."""
    if lazy_result and cache_result:
      raise ValueError(
          'The result of a traced call cannot be both lazy and cached.'
          f'calling: {value=}'
      )
    if cache_result:
      result = cls(value=None, _cache_result=True)
      # Direct insert to the cache without retrieving.
      result.result_.cache_insert(result, value)
      return result
    return cls(value=value, _lazy_result=lazy_result)

  @property
  def id(self) -> int:
    return self._id

  @property
  def cache_result(self):
    return self._cache_result

  @property
  def lazy_result(self):
    return self._lazy_result

  def __str__(self):
    if self.value is None:
      return f'LazyObject(id={self.id})'
    elif isinstance(self.value, type):
      return self.value.__name__
    else:
      return f'<{self.value.__class__.__name__} object>'

  def __hash__(self) -> int:
    if not self._cache_result:
      try:
        return hash(self.value)
      except TypeError:
        # Fall back to hashing on id when value failed.
        pass
    return hash(self.id)

  def __eq__(self, other: Self) -> bool:
    if self.id == other.id:
      return True
    if not self._cache_result and not other._cache_result:
      return self.value == other.value
    return False

  def set_(self, **kwargs):
    return dc.replace(self, **kwargs)

  @_maybe_lru_cache(maxsize=_LAZY_OBJECT_CACHE_SIZE)
  def result_(self) -> _T | LazyObject[_T]:
    """Dereference the lazy object."""
    result = self.value
    if self._lazy_result:
      result = LazyObject.new(result)
    return result

  # Overrides to support pickling when getattr is overridden.
  def __getstate__(self):
    return dict(self.__dict__)

  # Overrides to support pickling when getattr is overridden.
  def __setstate__(self, state):
    self.__dict__.update(state)

  # The following are remote builtin methods.
  def __call__(
      self,
      *args,
      cache_result_: bool = False,
      lazy_result_: bool = False,
      **kwargs,
  ) -> LazyFn:
    """Calling a LazyFn records a lazy result of the call."""
    if lazy_result_ and cache_result_:
      raise ValueError(
          'The result of a traced call cannot be both lazy and cached.'
          f'calling: {self} with {args=}, {kwargs=}'
      )
    return LazyFn.new(
        self,
        args=args,
        kwargs=kwargs,
        lazy_result=lazy_result_,
        cache_result=cache_result_,
    )

  def __getattr__(self, name) -> LazyFn:
    if name.startswith('__') and name.endswith('__'):
      raise AttributeError
    return LazyFn.new(getattr, args=(self, name))

  def __getitem__(self, key) -> LazyFn:
    return LazyFn.new(operator.getitem, args=(self, key))


@dc.dataclass(kw_only=True, frozen=True)
class LazyFn(LazyObject[_T]):
  """A lazy function that has all the information to be called later.

  Attributes:
    value: The function to be called.
    args: The positional arguments later used to pass in the fn.
    kwargs: The named arguments later used to pass in the fn.
    cache_result: If True, cache the result of the make LazyFn.
    lazy_result: If True, return a LazyObject instead of the actual result.
    id: The id of the LazyFn that persists across processes.
  """

  args: tuple[Hashable | LazyFn, ...] = ()
  kwargs: tuple[tuple[str, Hashable | LazyFn], ...] = ()

  @classmethod
  def new(
      cls,
      value: Callable[..., _T] | None = None,
      *,
      args: Sequence[Hashable] = (),
      kwargs: Mapping[str, Hashable] | None = None,
      cache_result: bool = False,
      lazy_result: bool = False,
  ) -> Self:
    """Normalizes the arguments before constructing a LazyFn."""
    return cls(
        value=value,
        args=tuple(args),
        kwargs=normalize_kwargs(kwargs or {}),
        _cache_result=cache_result,
        _lazy_result=lazy_result,
    )

  def __str__(self) -> str:
    if self.value is getattr:
      return f'{self.args[0]}.{self.args[1]}'
    elif self.value is operator.getitem:
      return f'{self.args[0]}[{self.args[1]}]'
    args = ', '.join(f'{arg}' for arg in self.args)
    kwargs = ', '.join(f'{k}={v}' for k, v in self.kwargs)
    args_strs = []
    if args:
      args_strs.append(args)
    if kwargs:
      args_strs.append(kwargs)
    args_str = ', '.join(args_strs)
    return f'{self.value}({args_str})'

  def __hash__(self) -> int:
    try:
      return hash((self.value, self.args, self.kwargs))
    except TypeError:
      return hash(self.id)

  def __eq__(self, other: Self):
    if self.id == other.id:
      return True
    return (
        self.value == other.value
        and self.args == other.args
        and self.kwargs == other.kwargs
    )

  @_maybe_lru_cache(maxsize=128)
  def result_(self):
    """Instantiate a lazy fn to a actual fn when applicable."""
    if self.value is None:
      result = None
    else:
      fn = _maybe_make(self.value)
      if not callable(fn):
        raise TypeError(f'fn is not callable from {self}.')
      args = tuple(_maybe_make(arg) for arg in self.args)
      kwargs = {k: _maybe_make(v) for k, v in self.kwargs}
      result = _maybe_make(fn(*args, **kwargs))
    if self._lazy_result:
      result = LazyObject.new(result)
    return result


def clear_cache():
  """Clear the cache for maybe_make."""
  LazyFn.result_.cache_clear()


def cache_info():
  """Returns the cache info for lazy_fn."""
  return LazyFn.result_.cache_info()


def object_info():
  """Returns the cache info for lazy_objects."""
  return LazyObject.result_.cache_info()


def clear_object():
  """Returns the cache info for lazy_objects."""
  return LazyObject.result_.cache_clear()


def is_resolvable(obj: Any):
  return (
      isinstance(obj, base_types.Resolvable) or makeables[type(obj)] is not None
  )


def trace(
    value: _T,
    *,
    use_cache: bool = False,
    lazy_result: bool = False,
) -> LazyObject[_T]:
  """Traces a callable to record the function and its arguments.

  A lazy function is the lazy counterpart of the actual function. We can convert
  a lazy function using `lazy`: fn -> lazy(fn) -> lazy_fn. Calling a lazy
  counterpart of the function doesn't call the actual function, but record the
  arguments used to call the function later. To call the actual function, use
  `chainable.maybe_make(lazy_fn)`. E.g.,
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
  The argument to the call() function is arbitrarily bindable; e.g.,
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
    value: The value to be dereferenced or called lazily.
    use_cache: Deprecated, uses cache_result instead.
    lazy_result: If True, return a LazyObject instead of the actual result.

  Returns:
    A function that records the fn and its arguments to be called later.
  """
  if use_cache:
    logging.warning(
        '`use_cache` is deprecated, please directly use `cache_result_` at the'
        ' callsite. E.g., trace(Model)(path="", cache_result_=True).'
        'traced value: %s',
        value,
    )
  # This purely just trace the value, which means the value is included in
  # the object. Unlike LazyObject.new(value) that stores the value locally
  # and can only be derefernced later.
  return LazyObject.new(value, cache_result=False, lazy_result=lazy_result)


# TODO: b/311207032 - Deprecate Makeable interface in favor of Resolvable.
@dc.dataclass(frozen=True)
class MakeableLazyFn(base_types.Makeable[_T]):
  """Wraps a LazyFn to be used as a Makeable."""

  lazy_fn: base_types.Resolvable[_T]

  def make(self) -> _T:
    return maybe_make(self.lazy_fn)
