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
"""Internal batching utils, not meant to be used by users."""

from collections.abc import Iterable, Iterator
import functools
import queue
from typing import Any, TypeVar

import more_itertools as mit
import numpy as np

_ValueT = TypeVar('_ValueT')


def _dequeue_as_generator(
    input_queue: queue.SimpleQueue[_ValueT | StopIteration],
) -> Iterator[_ValueT]:
  """Converts a queue to an iterator, stops when meeting StopIteration."""
  while not isinstance(value := input_queue.get(), StopIteration):
    yield value
  return value.value


class RecitableIterator(Iterator[_ValueT]):
  """An iterator that recite its inputs."""

  def __init__(self, iterator: Iterable[_ValueT], *, max_buffer_size: int = 0):
    self._iterator = iter(iterator)
    self._max_buffer_size = max_buffer_size
    self.buffer = queue.SimpleQueue()

  def __next__(self):
    try:
      value = next(self._iterator)
    except StopIteration as e:
      self.buffer.put(e)
      raise e
    self.buffer.put(value)
    if self._max_buffer_size and self.buffer.qsize() > self._max_buffer_size:
      raise ValueError(
          f'Buffer overflow: {self.buffer.qsize()} > {self._max_buffer_size=}.'
      )
    return value

  def __iter__(self):
    return self

  def recite_iterator(self) -> Iterator[_ValueT]:
    return _dequeue_as_generator(self.buffer)


def _concat(data: list[Any]):
  """Concatenates array like data."""
  if len(data) == 1:
    return data[0]
  (batch,), data = mit.spy(data)
  if hasattr(batch, '__array__'):
    return np.concatenate(list(data))
  elif isinstance(batch, list):
    return list(mit.flatten(data))
  elif isinstance(batch, tuple):
    return tuple(mit.flatten(data))
  else:
    raise TypeError(
        f'Unsupported container type: {type(batch)}, only list, tuple and numpy'
        ' array are supported.'
    )


def _batch_size(data: Any):
  if hasattr(data, '__array__'):
    return data.shape[0]
  else:
    try:
      return len(data)
    except TypeError as e:
      raise TypeError(
          f'Non sequence type: {type(data)}, only sequences are supported.'
      ) from e


def rebatched(
    data: Iterator[tuple[Any, ...]],
    batch_size: int = 0,
    *,
    num_columns: int,
):
  """Merges and concatenates n batches while iterating."""
  if not batch_size:
    yield from data

  column_buffer = [[] for _ in range(num_columns)]
  batch_sizes = np.zeros(num_columns, dtype=int)
  exhausted = False
  last_columns = None
  while not exhausted:
    if (batch := next(data, None)) is None:
      exhausted = True
    else:
      if len(batch) != num_columns:
        raise ValueError(
            f'Incorrect number of columns, got {len(batch)} != {num_columns=}.'
        )
      for i, column in enumerate(batch):
        column_buffer[i].append(column)
      batch_sizes += [_batch_size(column) for column in batch]
      if not all(batch_sizes[0] == each_size for each_size in batch_sizes):
        raise ValueError(
            f'Hetroegeneous columns number, got {batch_sizes=} does not equal'
            f' {batch_size=}.'
        )
    # Flush the buffer.
    if batch_sizes[0] and (batch_sizes[0] >= batch_size or exhausted):
      concated = map(_concat, column_buffer)
      sliced_by_batch_size = functools.partial(mit.sliced, n=batch_size)
      for columns in mit.zip_equal(*map(sliced_by_batch_size, concated)):
        # Only at most one batch can remain after slicing.
        if last_columns is not None:
          yield last_columns
        last_columns = columns
      column_buffer = [[] for _ in range(num_columns)]
      batch_sizes = np.zeros(num_columns, dtype=int)
      if last_columns is None:
        continue
      if exhausted or _batch_size(last_columns[0]) == batch_size:
        yield last_columns
      else:
        column_buffer = [[column] for column in last_columns]
        batch_sizes += [_batch_size(column) for column in last_columns]
      last_columns = None
