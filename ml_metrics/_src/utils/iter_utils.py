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

from collections.abc import Callable, Iterable, Iterator
import functools
import inspect
import queue
import time
from typing import Any, TypeVar, cast

from absl import logging
import more_itertools as mit
import numpy as np


_ValueT = TypeVar('_ValueT')
_InputT = TypeVar('_InputT')

STOP_ITERATION = StopIteration()


def is_stop_iteration(e: Exception) -> bool:
  return e is STOP_ITERATION or isinstance(e, StopIteration)


class PrefetchedIterator:
  """An iterator that can also prefetch before iterated."""

  def __init__(self, generator, prefetch_size: int = 2):
    self._data = queue.SimpleQueue()
    self._returned = None
    if not inspect.isgenerator(generator):
      generator = iter(generator)
    self._generator = generator
    self._exceptions = []
    self._prefetch_size = prefetch_size
    self._exhausted = False
    self._error_cnt = 0
    self._cnt = 0
    self._data_size = 0

  @property
  def cnt(self) -> int:
    return self._cnt

  @property
  def returned(self) -> Any:
    assert (
        self._exhausted
    ), 'Generator is not exhausted, returned is not available.'
    return self._returned

  @property
  def data_size(self) -> int:
    return self._data_size

  @property
  def exhausted(self) -> bool:
    return self._exhausted

  @property
  def exceptions(self) -> list[Exception]:
    return self._exceptions

  def flush_prefetched(self, batch_size: int = 0) -> list[Any]:
    """Flushes the prefetched data.

    Args:
      batch_size: the batch size of the data to be flushed. If batch_size = 0,
        it takes all prefetche immediately.

    Returns:
      The flushed data.
    """
    result = []
    while (
        self.data_size < batch_size
        and not self._exhausted
        and not self._exceptions
    ):
      time.sleep(0)
    batch_size = batch_size or self.data_size
    while self.data_size and len(result) < batch_size:
      result.append(self._data.get())
      self._data_size -= 1
    logging.info('Chainables: flush_prefetched: %s', len(result))
    return result

  def __next__(self):
    self.prefetch(1)
    if not self._data.empty():
      return self._data.get()
    else:
      logging.info('chainables: Generator exhausted from %s.', self._generator)
      raise StopIteration(self._returned)

  def __iter__(self):
    return self

  def prefetch(self, num_items: int = 0):
    """Prefeches items from the undelrying generator."""
    while not self._exhausted and self._data.qsize() < (
        num_items or self._prefetch_size
    ):
      try:
        self._data.put(next(self._generator))
        self._cnt += 1
        self._data_size += 1
      except StopIteration as e:
        self._exhausted = True
        self._returned = e.value
        logging.info(
            'chainables: prefetch exhausted after %d items.', self._cnt
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('chainables: Got error during prefetch.')
        if 'generator already executing' != str(e):
          self._exceptions.append(e)
          if len(self._exceptions) > 3:
            logging.exception('chainables: Too many errors, stop prefetching.')
            break

        time.sleep(0)


def enqueue_from_generator(
    generator: Iterator[_ValueT],
    output_queue: queue.Queue[_ValueT | StopIteration],
    timeout: float | None = None,
) -> Iterator[Any]:
  """Iterates through a generator while enqueue its elements."""
  generator = iter(generator)
  exhausted = False
  while not exhausted:
    try:
      value = next(generator)
    except StopIteration as e:
      value = e
      exhausted = True
    ticker = time.time()
    enqueued = False
    while not enqueued:
      try:
        output_queue.put_nowait(value)
      except queue.Full as e:
        time.sleep(0)
        if timeout is not None and time.time() - ticker > timeout:
          raise TimeoutError(f'Enqueue timeout after {timeout} seconds.') from e
        continue
      enqueued = True
      if is_stop_iteration(value):
        return cast(StopIteration, value).value
      yield value


def dequeue_as_generator(
    input_queue: queue.Queue[_ValueT | StopIteration],
    *,
    num_steps: int = -1,
    timeout: float | None = None,
) -> Iterator[_ValueT]:
  """Converts a queue to an iterator, stops when meeting StopIteration."""
  i = 0
  ticker = None
  run_until_exhausted = num_steps < 0
  while run_until_exhausted or i < num_steps:
    ticker = ticker or time.time()
    try:
      value = input_queue.get_nowait()
    except queue.Empty as e:
      if timeout is not None and ticker and time.time() - ticker > timeout:
        raise TimeoutError(f'Dequeue timeout after {timeout} seconds.') from e
      time.sleep(0)
      continue
    ticker = None  # Reset the ticker to indicate the last get() is successful.
    if is_stop_iteration(value):
      return cast(StopIteration, value).value
    yield value
    i += 1


def _dequeue_as_generator_blocking(
    input_queue: queue.SimpleQueue[_ValueT | StopIteration],
) -> Iterator[_ValueT]:
  """Converts a queue to an iterator, stops when meeting StopIteration."""
  while not is_stop_iteration(value := input_queue.get()):
    yield value
  return cast(StopIteration, value).value


class _RecitableIterator(Iterator[_ValueT]):
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
    return _dequeue_as_generator_blocking(self.buffer)


def processed_with_inputs(
    process_fn: Callable[[Iterable[_InputT]], Iterable[_ValueT]],
    input_iterator: Iterable[_InputT],
    *,
    max_buffer_size: int = 0,
) -> Iterator[tuple[_InputT, _ValueT]]:
  """Zips the processed outputs with its inputs."""
  iter_input = _RecitableIterator(
      input_iterator, max_buffer_size=max_buffer_size
  )
  iter_output = process_fn(iter_input)
  # Note that recital iterator has to be put after the input iterator so that
  # there are values to be recited.
  return zip(iter_output, iter_input.recite_iterator())


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


def rebatched_tuples(
    tuples: Iterator[tuple[_ValueT, ...]],
    batch_size: int = 0,
    *,
    num_columns: int,
) -> Iterator[tuple[_ValueT, ...]]:
  """Merges and concatenates n batches while iterating."""
  if not batch_size:
    yield from tuples
    return

  assert num_columns > 0
  column_buffer = [[] for _ in range(num_columns)]
  batch_sizes = np.zeros(num_columns, dtype=int)
  exhausted = False
  last_columns = None
  while not exhausted:
    if (batch := next(tuples, None)) is None:
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
    # Flush the buffer when the batch size is reached.
    has_batch_sizes = batch_sizes.size and batch_sizes[0]
    if has_batch_sizes and (batch_sizes[0] >= batch_size or exhausted):
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
