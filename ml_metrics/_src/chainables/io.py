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
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
import copy
import dataclasses as dc
from typing import Self, TypeVar

from ml_metrics._src import types

_T = TypeVar('_T')


@dc.dataclass(frozen=True, slots=True)
class ShardConfig:
  shard_index: int = 0
  num_shards: int = 1
  start_index: int = 0


@dc.dataclass(frozen=True)
class ShardedSequence(types.Recoverable, Iterable[_T]):
  """A sharded data source for chainables."""
  data: Sequence[_T]
  _shard_state: ShardConfig = dc.field(default_factory=ShardConfig)

  def __post_init__(self):
    data = self.data
    if not hasattr(data, '__getitem__') or not hasattr(data, '__len__'):
      raise TypeError(f'data is not indexable, got {type(data)=}')
    if self._shard_state.num_shards < 1:
      raise ValueError(f'num_shards must be positive, got {self._shard_state=}')

  def shard(self, shard_index: int, num_shards: int) -> Self:
    return dc.replace(self, _shard_state=ShardConfig(shard_index, num_shards))

  @property
  def state(self) -> ShardConfig:
    return self._shard_state

  def from_state(self, shard_state: ShardConfig) -> Self:
    """Iterates the data source given a shard index."""
    return dc.replace(self, _shard_state=shard_state)

  def iterate(self) -> _SequenceIterator[_T]:
    return _SequenceIterator(self)

  def __iter__(self) -> _SequenceIterator[_T]:
    return self.iterate()


class _SequenceIterator(types.Recoverable, Iterator[_T]):
  """A sharded data source for chainables."""

  config: ShardedSequence
  _data: Sequence[_T]
  _start_index: int
  _index: int
  _end_index: int

  def __init__(self, config: ShardedSequence):
    self.config = config
    self._data = copy.deepcopy(config.data)
    shard_state = config.state
    shard_index, num_shards = shard_state.shard_index, shard_state.num_shards
    interval, remainder = divmod(len(self._data), num_shards)
    start, adjusted_interval = 0, 0
    for i in range(shard_index + 1):
      adjusted_interval = interval + 1 if i < remainder else interval
      start += adjusted_interval if i < shard_index else 0
    self._index = self._start_index = start + shard_state.start_index
    self._end_index = start + adjusted_interval
    self.shard_state = shard_state

  def from_state(self, shard_state: ShardConfig) -> Self:
    return self.__class__(self.config.from_state(shard_state))

  @property
  def state(self) -> ShardConfig:
    start_index = self._index - self._start_index
    return dc.replace(self.shard_state, start_index=start_index)

  def __next__(self) -> _T:
    """Iterates the data source given a shard index."""
    if self._index >= self._end_index:
      raise StopIteration
    result = self._data[self._index]
    self._index += 1
    return result

  def __iter__(self) -> Self:
    """Iterates the data source given a shard index."""
    return self


@dc.dataclass(frozen=True)
class ShardedIterable(types.Recoverable, Iterable[_T]):
  """A sharded data source for any iterable."""
  data: Iterable[_T]
  _shard_state: ShardConfig = dc.field(default_factory=ShardConfig)

  def __post_init__(self):
    data = self.data
    if not (isinstance(data, Iterable) and not isinstance(data, Iterator)):
      raise TypeError(
          f'input has to be an iterable but not an iterator, got {type(data)=}'
      )
    if self._shard_state.num_shards < 1:
      raise ValueError(f'num_shards must be positive, got {self._shard_state=}')

  def shard(self, shard_index: int, num_shards: int) -> Self:
    return dc.replace(self, _shard_state=ShardConfig(shard_index, num_shards))

  @property
  def state(self) -> ShardConfig:
    return self._shard_state

  def from_state(self, shard_state: ShardConfig) -> Self:
    return dc.replace(self, _shard_state=shard_state)

  def iterate(self) -> _ResumableIterator[_T]:
    return _ResumableIterator(self)

  def __iter__(self) -> _ResumableIterator[_T]:
    return self.iterate()


class _ResumableIterator(types.Recoverable, Iterator[_T]):
  """An sharded iterator for an iterable."""

  def __init__(self, config: ShardedIterable):
    self.config = config
    self._index = 0
    self._it = iter(config.data)

  def from_state(self, shard_state: ShardConfig) -> Self:
    return self.__class__(self.config.from_state(shard_state))

  @property
  def state(self) -> ShardConfig:
    return dc.replace(self.config.state, start_index=self._index)

  def __next__(self) -> _T:
    """Iterates the data source given a shard index."""
    while self._index < self.config.state.start_index:
      _ = next(self._it)
      self._index += 1
    shard_index = self.config.state.shard_index
    num_shards = self.config.state.num_shards
    while self._index % num_shards != shard_index:
      _ = next(self._it)
      self._index += 1
    result = next(self._it)
    self._index += 1
    return result

  def __iter__(self) -> Self:
    return self
