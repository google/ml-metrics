# Copyright 2025 Google LLC
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
"""I/O utilities for chainables."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Self, TypeVar
from ml_metrics._src import types

_T = TypeVar('_T')


class ShardedSequence(types.Shardable, types.Serializable, Iterable[_T]):
  """A sharded data source for chainables."""

  def __init__(self, data: Sequence[_T]):
    if not hasattr(data, '__getitem__') or not hasattr(data, '__len__'):
      raise TypeError(f'data is not indexable, got {type(data)=}')
    self._data_len = len(data)
    self._data = data
    self._start_index = 0

  @classmethod
  def from_state(cls, state: Sequence[_T]) -> Self:
    return cls(state)

  def get_state(self) -> Sequence[_T]:
    return self._data[self._start_index :]

  def get_shard(self, shard_index: int, num_shards: int = 0) -> Iterable[_T]:
    """Iterates the data source given a shard index."""
    if num_shards < 1:
      raise ValueError(f'num_shards must be positive, got {num_shards=}')
    interval, remainder = divmod(self._data_len - self._start_index, num_shards)
    start, adjusted_interval = self._start_index, 0
    for i in range(shard_index + 1):
      adjusted_interval = interval + 1 if i < remainder else interval
      start += adjusted_interval if i < shard_index else 0
    end = start + adjusted_interval
    print('sharding: ', start, end)
    return self.__class__(self._data[start: end])

  def __next__(self) -> _T:
    """Iterates the data source given a shard index."""
    if self._start_index >= self._data_len:
      raise StopIteration
    result = self._data[self._start_index]
    self._start_index += 1
    return result

  def __iter__(self) -> Iterator[_T]:
    """Iterates the data source given a shard index."""
    return self


class ShardedIterable(types.Shardable, Iterable[_T]):
  """A sharded data source for chainables."""

  def __init__(
      self, data: Iterable[_T], *, shard_index: int = 0, num_shards: int = 1
  ):
    if not (isinstance(data, Iterable) and not isinstance(data, Iterator)):
      raise TypeError(
          f'input has to be an iterable but not an iterator, got {type(data)=}'
      )
    if num_shards < 1:
      raise ValueError(f'num_shards must be positive, got {num_shards=}')
    self._data = data
    self._shard_index = shard_index
    self._num_shards = num_shards
    self._index = -1

  @property
  def num_shards(self) -> int:
    return self._num_shards

  def __iter__(self) -> Iterator[_T]:
    """Iterates the data source given a shard index."""
    for i, value in enumerate(self._data):
      if i % self._num_shards == self._shard_index:
        self._index += 1
        yield value

  def get_shard(self, shard_index: int, num_shards: int = 0) -> Self:
    return ShardedIterable(
        self._data,
        shard_index=shard_index,
        num_shards=num_shards or self._num_shards,
    )
