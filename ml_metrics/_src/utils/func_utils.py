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
from collections.abc import Iterator, Mapping
import dataclasses as dc
from typing import TypeVar

_KeyT = TypeVar('_KeyT')
_ValueT = TypeVar('_ValueT')


@dc.dataclass(slots=True, kw_only=True, frozen=True)
class _CacheInfo:
  hits: int
  misses: int
  maxsize: int
  currsize: int


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
