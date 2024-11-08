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
"""Internal function utils, not meant to be used by users."""

import collections
from collections.abc import Iterable, Iterator, Mapping
import copy
import dataclasses as dc
import functools
import itertools as itt
from typing import TypeVar
import weakref

from absl import logging
import more_itertools as mit


_KeyT = TypeVar('_KeyT')
_ValueT = TypeVar('_ValueT')
_T = TypeVar('_T')


@dc.dataclass(slots=True, frozen=True)
class _CacheInfo:
  hits: int
  misses: int
  currsize: int
  maxsize: int = 0


class LruCache(Mapping[_KeyT, _ValueT]):
  """A mapping like object for caching with limited size."""

  def __init__(self, maxsize=128):
    self.maxsize = maxsize
    self.currsize = 0
    self.hits = 0
    self.misses = 0
    self.data = collections.OrderedDict()

  def __getitem__(self, key):
    if key not in self.data:
      self.misses += 1
      raise KeyError()
    self.hits += 1
    value = self.data[key]
    self.data.move_to_end(key)
    return value

  def __setitem__(self, key, value):
    key_is_new = key not in self.data
    self.data[key] = value
    if key_is_new:
      self.currsize += 1
      self.data.move_to_end(key)
    if self.currsize > self.maxsize:
      oldest = next(iter(self.data))
      del self.data[oldest]
      self.currsize -= 1

  def cache_insert(self, key, value):
    self.__setitem__(key, value)

  def __contains__(self, key):
    return key in self.data

  def __iter__(self) -> Iterator[_KeyT]:
    return iter(self.data)

  def __len__(self) -> int:
    return self.currsize

  def cache_clear(self):
    self.data.clear()
    self.currsize = 0
    self.hits = 0
    self.misses = 0

  def cache_info(self) -> _CacheInfo:
    return _CacheInfo(
        hits=self.hits,
        misses=self.misses,
        maxsize=self.maxsize,
        currsize=self.currsize,
    )


def lru_cache(
    fn=None,
    *,
    settable_kwargs: Iterable[str] = (),
    maxsize: int = 128,
):
  """Cache by the positional and specified keyword arguments."""

  settable_kwargs = set(settable_kwargs)

  def decorator(fn):
    cache_ = LruCache(maxsize=maxsize)

    @functools.wraps(fn)
    def wrapped(*args, cache_insert_: bool = False, **kwargs):
      is_settable = lambda x: x[0] in settable_kwargs
      hashed, settables = mit.partition(is_settable, kwargs.items())
      key = hash(tuple(itt.chain(args, hashed)))
      if not cache_insert_ and key in cache_:
        result = cache_[key]
      else:
        logging.debug('chainable: cache miss %s: %s, %s', fn, args, kwargs)
        result = fn(*args, **kwargs)
        cache_[key] = result
      result_new = None
      for k, v in settables:
        if v != getattr(result, k):
          if result_new is None:
            result_new = copy.copy(result)
          setattr(result_new, k, v)
      return result_new if result_new is not None else result

    wrapped.cache_info = cache_.cache_info
    wrapped.cache_clear = cache_.cache_clear
    return wrapped

  return decorator if fn is None else decorator(fn)


class SingletonMeta(type):
  """A metaclass that makes a class a singleton of any "equivalent" instance.

  The actual class instance has to be hashable to test equivalence. This also
  works with inherited classes.
  Example:
  ```
  @dc.dataclass(frozen=True)
  class Foo(metaclass=SingletonMeta):
    a: int
    b: str = 'b'

  # The following should be true.
  assert Foo(1) is Foo(1)
  assert Foo(1, 'b') is Foo(1)
  assert Foo(1, b='b') is not Foo(1)
  assert Foo(2) is not Foo(1)
  assert Foo(1, 'a') is not Foo(1)
  ```
  """

  _instances = weakref.WeakKeyDictionary()

  def __call__(cls: type[_T], *args, **kwargs) -> _T:
    obj = super(SingletonMeta, cls).__call__(*args, **kwargs)
    if obj in cls._instances and (result := cls._instances[obj]()) is not None:
      return result
    logging.info('%s', f'chainable: singleton {cls.__name__}, {obj}')
    cls._instances[obj] = weakref.ref(obj)
    return obj
