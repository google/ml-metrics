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
import inspect
from typing import Any, Generic, TypeVar

ValueT = TypeVar('ValueT')
Fn = Callable[..., ValueT]


class _Makers(collections.UserDict):
  """Maker registry."""

  def register(self, type_: type[Any], maker: Callable[..., Any]):
    self.data[repr(type_)] = maker

  def __getitem__(self, type_: type[Any]) -> Callable[..., Any] | None:
    return self.data.get(repr(type_), None)


makeables = _Makers()


def maybe_make(maybe_lazy: Any) -> Any:
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
class LazyFn(Generic[ValueT], Callable[..., 'LazyFn']):
  """A lazy function that has all the information to be called later.

  Attributes:
    fn: The function to be called.
    args: The positional arguements later used to pass in the fn.
    kwargs: The named arguements later used to pass in the fn.
  """

  fn: Callable[..., ValueT] | LazyFn | None = None
  args: tuple[Hashable | LazyFn, ...] = ()
  kwargs: tuple[tuple[str, Hashable | LazyFn], ...] = ()

  @classmethod
  def new(
      cls,
      fn: Callable[..., ValueT] | None = None,
      *,
      args: Sequence[Hashable] = (),
      kwargs: Mapping[str, Hashable] | None = None,
  ) -> LazyFn[ValueT]:
    """Normalizes the arguements before constructing a LazyFn."""
    kwargs = (kwargs or {}).items()
    return cls(fn=fn, args=tuple(args), kwargs=tuple(kwargs))

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


def _make(fn: LazyFn) -> Any:
  """Instantiate a lazy fn to a actual fn when applicable."""
  assert isinstance(fn, LazyFn), f'{fn} is not a LazyFn, instead: {type(fn)}.'
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


makeables.register(LazyFn, _make)


def trace(fn: Callable[..., ValueT]) -> Callable[..., LazyFn[ValueT]]:
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

  Returns:
    A function that records the fn and its arguements to be called later.
  """

  def wrapped(*args, **kwargs):
    return LazyFn.new(fn=fn, args=args, kwargs=kwargs)

  return wrapped
