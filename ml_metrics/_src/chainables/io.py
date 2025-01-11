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
from typing import TypeVar
from ml_metrics._src import types

_T = TypeVar('_T')


class ShardedSequence(types.Shardable, Iterable[_T]):
  """A sharded data source for chainables."""

  def __init__(
      self, data: Sequence[_T], *, shard_index: int = 0, num_shards: int = 1
  ):
    if not hasattr(data, '__getitem__') or not hasattr(data, '__len__'):
      raise TypeError(f'data is not indexable, got {type(data)=}')
    if num_shards < 1:
      raise ValueError(f'num_shards must be positive, got {num_shards=}')
    self._data_len = len(data)
    self._data = data
    self._shard_index = shard_index
    self._num_shards = num_shards

  @property
  def num_shards(self) -> int:
    return self._num_shards

  def get_shard(self, shard_index: int, num_shards: int = 0) -> Iterable[_T]:
    """Iterates the data source given a shard index."""
    return self.__class__(
        self._data,
        shard_index=shard_index,
        num_shards=num_shards or self._num_shards,
    )

  def __iter__(self) -> Iterator[_T]:
    """Iterates the data source given a shard index."""
    interval, remainder = divmod(self._data_len, self._num_shards)
    start, adjusted_interval = 0, 0
    for i in range(self._shard_index + 1):
      adjusted_interval = interval + 1 if i < remainder else interval
      start += adjusted_interval if i < self._shard_index else 0
    for value in self._data[start : start + adjusted_interval]:
      yield value
