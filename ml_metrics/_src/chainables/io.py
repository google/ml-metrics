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

from collections.abc import Iterable, Iterator
import dataclasses as dc
from typing import Any, Self, TypeVar

from ml_metrics._src import types
from ml_metrics._src.utils import iter_utils

_T = TypeVar('_T')


@dc.dataclass(frozen=True, slots=True)
class ShardConfig:
  shard_index: int = 0
  num_shards: int = 1
  start_index: int = 0
  parent: ShardConfig | None = dc.field(default=None, kw_only=True)


def maybe_shardable(data_source: _T) -> types.Shardable | _T:
  """Returns a shardable data source if possible."""
  if types.is_shardable(data_source):
    return data_source

  if types.is_random_accessible(data_source):
    return SequenceDataSource(data_source)

  return data_source


@dc.dataclass(frozen=True)
class SequenceDataSource(types.Recoverable, Iterable[_T]):
  """A shardable sequence data source."""
  data: types.RandomAccessible[_T]
  ignore_error: bool = dc.field(kw_only=True, default=False)
  batch_size: dc.InitVar[int] = 0
  _shard_state: ShardConfig = dc.field(default_factory=ShardConfig)
  _start: int = 0
  _end: int | None = None

  def __post_init__(self, batch_size: int):
    data = self.data
    if not hasattr(data, '__getitem__') or not hasattr(data, '__len__'):
      raise TypeError(f'data is not indexable, got {type(data)=}')
    # Use MergedSequences even for a single sequence to enforce iterating by
    # random access so that the iterator is continuable after exception.
    sequences = [self.data]
    if isinstance(self.data, iter_utils.MergedSequences):
      sequences = self.data.sequences
    data = iter_utils.MergedSequences(sequences, batch_size)
    object.__setattr__(self, 'data', data)

  @classmethod
  def from_sequences(
      cls,
      sequences: Iterable[types.RandomAccessible[_T]],
      batch_size: int = 0,
      ignore_error: bool = False,
  ) -> Self:
    return cls(
        iter_utils.MergedSequences(sequences),
        ignore_error=ignore_error,
        batch_size=batch_size,
    )

  def shard(self, shard_index: int, num_shards: int, offset: int = 0) -> Self:
    if num_shards < 1:
      raise ValueError(f'num_shards must be positive, got {num_shards=}')
    interval, remainder = divmod(self.end - self.start, num_shards)
    start, adjusted_interval = self.start, 0
    for i in range(shard_index + 1):
      adjusted_interval = interval + 1 if i < remainder else interval
      start += adjusted_interval if i < shard_index else 0
    shard_state = ShardConfig(
        shard_index, num_shards, offset, parent=self._shard_state
    )
    return dc.replace(
        self,
        _shard_state=shard_state,
        _start=start + offset,
        _end=start + adjusted_interval,
    )

  @property
  def start(self) -> int:
    return self._start

  @property
  def end(self) -> int:
    return len(self.data) if self._end is None else self._end

  def __len__(self) -> int:
    return self.end - self.start

  @property
  def state(self) -> ShardConfig:
    return self._shard_state

  def from_state(self, shard_state: ShardConfig) -> Self:
    """Iterates the data source given a shard index."""
    if shard_state.parent is not None:
      result = self.from_state(shard_state.parent)
    else:
      result = SequenceDataSource(self.data, ignore_error=self.ignore_error)
    return result.shard(
        shard_state.shard_index, shard_state.num_shards, shard_state.start_index
    )

  def iterate(self) -> SequenceIterator[_T]:
    return SequenceIterator(self)

  def __iter__(self) -> Iterator[_T]:
    return self.iterate()

  def __getitem__(self, index: int | Any) -> _T:
    """Iterates the data source given a shard index."""
    if isinstance(index, slice):
      start, stop, step = index.start, index.stop, index.step
      start = start or 0  # Convert None to 0.
      start += self.start
      start = min(start, self.end)
      if stop is None:
        stop = self.end
      else:
        stop = self.end + stop if stop < 0 else self.start + stop
        stop = min(stop, self.end)
      return list(self.data[slice(start, stop, step)])

    index = self.end + index if index < 0 else self.start + index
    return self.data[index]


class SequenceIterator(types.Recoverable, Iterator[_T]):
  """A sharded data source for chainables."""

  config: SequenceDataSource
  _index: int

  def __init__(self, config: SequenceDataSource):
    self._index = config.start
    iter_ = iter_utils.iter_ignore_error if config.ignore_error else iter
    self._it = iter_(config.data[config.start : config.end])
    self.config = config

  def from_state(self, shard_state: ShardConfig) -> Self:
    return self.__class__(self.config.from_state(shard_state))

  @property
  def state(self) -> ShardConfig:
    start_index = self._index - self.config.start
    return dc.replace(self.config.state, start_index=start_index)

  def __next__(self) -> _T:
    """Iterates the data source given a shard index."""
    result = next(self._it)
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

  def iterate(self) -> DataIterator[_T]:
    return DataIterator(self)

  def __iter__(self) -> Iterator[_T]:
    return self.iterate()


class DataIterator(types.Recoverable, Iterator[_T]):
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
