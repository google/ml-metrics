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
"""Internal iterator utils, not meant to be public."""

from __future__ import annotations

import abc
import asyncio
import collections
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Sequence
from concurrent import futures
import dataclasses as dc
import functools
import queue
import threading
import time
from typing import Any, Generic, Protocol, Self, TypeVar

from absl import logging
import more_itertools as mit
import numpy as np


_ValueT = TypeVar('_ValueT')
_InputT = TypeVar('_InputT')


class _QueueLike(Protocol[_ValueT]):
  """Protocol of the same interfaces of queue.Queue."""

  def get_nowait(self) -> _ValueT:
    """Similar to queue.Queue().get_nowait."""

  def put_nowait(self, value: _ValueT):
    """Similar to queue.Queue().put_nowait."""


STOP_ITERATION = StopIteration()


class SequenceArray(Sequence):
  """Impelements Python Sequence interfaces for a numpy array.

  This is a drop-in replacement whenever np.tolist() is called.
  Usage:
    x = SequenceArray(np.zeros((3,3)))
    # Use it as normal numpy array,
    x[2, 0] = 3
    # But also support Sequence interfaces.
    x.count((0, 0))
    x.index((0, 0))

  Attr:
    data: The underlying numpy array.
  """

  def __init__(self, value):
    self.data = value

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, i):
    return self.data[i]

  def __setitem__(self, i, v):
    self.data[i] = v

  def __iter__(self):
    return iter(self.data)

  def count(self, value):
    return mit.ilen(filter(lambda x: np.array_equal(x, value), self.data))

  def index(self, value):
    ixs = (i for i, elem in enumerate(self.data) if np.array_equal(elem, value))
    if (result := mit.first(ixs, None)) is None:
      raise ValueError(f'{value} is not in array')
    return result


@dc.dataclass(slots=True, eq=False)
class Progress:
  cnt: int = 0


class IterableQueue(Generic[_ValueT], abc.ABC):
  """Base implementation and interfaces for an (Async)Iterables queue."""

  @abc.abstractmethod
  def get(self) -> _ValueT:
    """Dequeue an element from the queue."""

  def dequeue_as_iterator(self, num_steps: int = -1) -> Iterator[_ValueT]:
    return _DequeueIterator(self, num_steps=num_steps)

  def __iter__(self):
    return self.dequeue_as_iterator()


class AsyncIterableQueue(IterableQueue[_ValueT]):
  """Base implementation and interfaces for an (Async)Iterables queue."""

  @abc.abstractmethod
  async def async_get(self):
    """Gets an element from the queue."""

  def async_dequeue_as_iterator(
      self, num_steps: int = -1
  ) -> AsyncIterator[_ValueT]:
    """Converts a queue to an iterator, stops when meeting StopIteration."""
    return _AsyncDequeueIterator(self, num_steps=num_steps)

  def __aiter__(self):
    return self.async_dequeue_as_iterator()


def _async_get(iterator_queue: IterableQueue):
  try:
    return iterator_queue.get()
  except StopIteration as e:
    raise StopAsyncIteration(*e.args) from e


class IteratorQueue(IterableQueue[_ValueT]):
  """Enqueue elements from an iterator and record exhausted and returned."""

  timeout: float | None
  name: str
  _queue: _QueueLike[_ValueT]
  # The lock here manages enqueue (put) and dequeue (get).
  _enqueue_lock: threading.Condition
  _dequeue_lock: threading.Condition
  # The lock here protects the access to all states below.
  _states_lock: threading.RLock
  _iterator_cnt: int | None
  _returned: list[Any]
  _exception: Exception | None
  _progress: Progress
  _exhausted: bool

  def __init__(
      self,
      queue_or_size: int | _QueueLike[_ValueT] = 0,
      *,
      name: str = '',
      timeout: float | None = None,
  ):
    if isinstance(queue_or_size, int):
      self._queue = self._default_queue(queue_or_size)
    else:
      self._queue = queue_or_size
    self.name = name
    self.timeout = timeout
    self._dequeue_lock = threading.Condition()
    self._enqueue_lock = threading.Condition()
    self._states_lock = threading.RLock()
    self._progress = Progress()
    self._returned = []
    self._exception = None
    self._exhausted = False
    self._iterator_cnt = None

  @classmethod
  def _default_queue(cls, maxsize: int) -> _QueueLike[_ValueT]:
    if maxsize == 0:
      return queue.SimpleQueue()
    return queue.Queue(maxsize=maxsize)

  @classmethod
  def from_queue(
      cls,
      q_or_size: int | _QueueLike[_ValueT],
      name='',
      timeout: float | None = None,
  ) -> Self:
    return cls(q_or_size, name=name, timeout=timeout)

  @property
  def exhausted(self) -> bool:
    return self._exhausted

  def __bool__(self):
    return not self.exhausted

  @property
  def exception(self) -> Exception | None:
    return self._exception

  @property
  def enqueue_done(self) -> bool:
    """Indicates whether there is ongoing enqueuer."""
    return self.exception is not None or self._iterator_cnt == 0

  @property
  def returned(self) -> list[Any]:
    return self._returned

  @property
  def progress(self) -> Progress:
    return self._progress

  def get_nowait(self) -> _ValueT:
    """Gets an element from the queue, raises Empty immediately if empty."""
    self._states_lock.acquire()
    try:
      return self._queue.get_nowait()
    except (queue.Empty, asyncio.QueueEmpty) as e:
      if self.enqueue_done:
        logging.debug(
            'chainable: %s dequeue stopped with_exception=%s, remaining %d',
            self.name,
            self.exception is not None,
            self._iterator_cnt,
        )
        self._exhausted = True
        raise self.exception or StopIteration(*self.returned) from e
      raise e
    except StopIteration as e:
      raise e
    except Exception as e:  # pylint: disable=broad-exception-caught
      e.add_note(f'Exception during dequeueing "{self.name}".')
      logging.exception('chainable: "%s" dequeue failed.', self.name)
      raise e
    finally:
      self._states_lock.release()

  def get(self) -> _ValueT:
    """Gets an element from the queue, waits for timeout if empty."""
    with self._dequeue_lock:
      while True:
        try:
          value = self.get_nowait()
          self._dequeue_lock.release()
          try:
            with self._enqueue_lock:
              self._enqueue_lock.notify()
          finally:
            self._dequeue_lock.acquire()
          logging.debug('chainable: "%s" dequeued a %s', self.name, type(value))
          return value
        except (queue.Empty, asyncio.QueueEmpty) as e:
          logging.debug('chainable: "%s" dequeue empty, waiting', self.name)
          if self._dequeue_lock.wait(timeout=self.timeout):
            logging.debug('chainable: "%s" dequeue retry', self.name)
            continue
          raise TimeoutError(f'Dequeue timeout={self.timeout}secs.') from e

  def put_nowait(self, value: _ValueT) -> None:
    """Puts a value to the queue, raieses if queue is full immediately."""
    try:
      self._queue.put_nowait(value)
    except (queue.Full, asyncio.QueueFull) as e:
      raise e
    with self._states_lock:
      self._progress.cnt += 1
      logging.debug(
          'chainable: "%s" enqueued cnt %d.', self.name, self._progress.cnt
      )

  def put(self, value: _ValueT) -> None:
    """Puts a value to the queue, waits for timeout if queue is full."""
    with self._enqueue_lock:
      while True:
        try:
          self.put_nowait(value)
          self._enqueue_lock.release()
          try:
            with self._dequeue_lock:
              self._dequeue_lock.notify()
          finally:
            self._enqueue_lock.acquire()
          break
        except (queue.Full, asyncio.QueueFull) as e:
          logging.debug('chainable: %s enqueue full, waiting', self.name)
          if self._enqueue_lock.wait(timeout=self.timeout):
            continue
          raise TimeoutError(f'Enqueue timeout={self.timeout}secs.') from e

  def _start_enqueue(self):
    with self._states_lock:
      if self._iterator_cnt is None:
        self._iterator_cnt = 0
      self._iterator_cnt += 1

  def _stop_enqueue(self, *values):
    """Stops enqueueing and records the returned values."""
    enqueue_done = False
    with self._states_lock:
      self._iterator_cnt -= 1
      if self._iterator_cnt < 0:
        raise RuntimeError(f'{self._iterator_cnt=} cannot be negative.')
      self._returned.extend(values)
      logging.debug(
          'chainable: "%s" enqueue stop, remaining %d, with_exception=%s',
          self.name,
          self._iterator_cnt,
          self.exception is not None,
      )
      if self.enqueue_done:
        enqueue_done = True
    if enqueue_done:
      # Unblock dequeue since there is no more values to dequeue.
      with self._dequeue_lock:
        logging.debug('chainable: %s enqueue done, notify all', self.name)
        self._dequeue_lock.notify_all()

  def enqueue_from_iterator(self, iterator: Iterable[_ValueT]):
    """Iterates through a generator while enqueue its elements."""
    iterator = iter(iterator)
    self._start_enqueue()
    while True:
      try:
        self.put(next(iterator))
      except StopIteration as e:
        self._stop_enqueue(*e.args)
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        e.add_note(f'Exception during enqueueing "{self.name}".')
        logging.exception('chainable: "%s" enqueue failed.', self.name)
        with self._states_lock:
          self._exception = e
        self._stop_enqueue()
        raise e


class _DequeueIterator:
  """An iterator that dequeues from an IteratorQueue."""

  def __init__(self, q: IterableQueue, *, num_steps: int = -1):
    self._iterator_queue = q
    self._num_steps = num_steps
    self._cnt = 0
    self._run_until_exhausted = num_steps < 0

  def __next__(self):
    if not self._run_until_exhausted and self._cnt == self._num_steps:
      raise StopIteration()
    value = self._iterator_queue.get()
    self._cnt += 1
    return value

  def __iter__(self):
    return self


class AsyncIteratorQueue(IteratorQueue[_ValueT], AsyncIterableQueue[_ValueT]):
  """A queue that can enqueue from and dequeue to an iterator."""
  _thread_pool: futures.ThreadPoolExecutor | None

  def __init__(
      self,
      queue_or_size: int | _QueueLike[_ValueT] = 0,
      *,
      name: str = '',
      timeout: float | None = None,
      thread_pool: futures.ThreadPoolExecutor | None = None,
  ):
    super().__init__(
        queue_or_size=queue_or_size,
        name=name,
        timeout=timeout,
    )
    self._thread_pool = thread_pool

  @classmethod
  def _default_queue(cls, maxsize: int):
    return asyncio.Queue(maxsize=maxsize)

  async def async_get(self):
    """Gets an element from the queue."""
    logging.debug('chainable: %s async dequeueing', self.name)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self._thread_pool, _async_get, self)

  async def async_put(self, value: _ValueT):
    logging.debug('chainable: %s async enqueueing a %s', self.name, type(value))
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self._thread_pool, self.put, value)

  async def async_enqueue_from_iterator(self, iterator: AsyncIterable[_ValueT]):
    """Iterates through a generator while enqueue its elements."""
    if not isinstance(iterator, AsyncIterator):
      iterator = aiter(iterator)
    self._start_enqueue()
    while True:
      try:
        value = await asyncio.wait_for(anext(iterator), self.timeout)
        await self.async_put(value)
      except StopAsyncIteration as e:
        return self._stop_enqueue(*e.args)
      except Exception as e:  # pylint: disable=broad-exception-caught
        e.add_note(f'Exception during async enqueueing {self.name}')
        logging.exception('chainable: %s enqueue failed.', self.name)
        with self._states_lock:
          self._exception = e
        self._stop_enqueue()
        raise e


class _AsyncDequeueIterator:
  """An async iterator that dequeues from an AsyncIteratorQueue."""

  def __init__(self, q: AsyncIterableQueue, *, num_steps: int = -1):
    self._iterator_queue = q
    self._num_steps = num_steps
    self._cnt = 0
    self._run_until_exhausted = num_steps < 0

  async def __anext__(self):
    if self._run_until_exhausted or self._cnt < self._num_steps:
      # AsyncIteratorQueue.get() can raise StopAsyncIteration.
      value = await self._iterator_queue.async_get()
      self._cnt += 1
      return value
    else:
      raise StopAsyncIteration()

  def __aiter__(self):
    return self


def piter(
    iterator_fn: Callable[..., Iterable[_ValueT]],
    input_iterator: Iterable[Any] | None = None,
    *,
    max_parallism: int = 8,
    buffer_size: int = 64,
    thread_pool: futures.ThreadPoolExecutor | None = None,
) -> Iterable[_ValueT]:
  """Call a chain of functions in sequence concurrently with multithreads.

  Args:
    iterator_fn: A generator function that takes an iterable as input.
    input_iterator: The input iterator to be passed to the iterator_fn.
    max_parallism: The maximum number of threads to be used.
    buffer_size: The buffer size of the queue.
    thread_pool: The thread pool to be used. If None, a new thread pool will be
      created.

  Returns:
    An iterable that iterates through the chain of functions.
  """
  output_q = IteratorQueue(buffer_size)
  input_q = None
  pool = thread_pool or futures.ThreadPoolExecutor()
  if input_iterator is not None:
    input_q = IteratorQueue(buffer_size)
    pool.submit(input_q.enqueue_from_iterator, input_iterator)
  for _ in range(max_parallism):
    if input_q is not None:
      it = iterator_fn(input_q)
    else:
      it = iterator_fn()
    pool.submit(output_q.enqueue_from_iterator, it)
  return output_q


def pmap(
    fn: Callable[..., Any],
    input_iterator: Iterable[Any] = (),
    *,
    max_parallism: int = 8,
    buffer_size: int = 64,
    thread_pool: futures.ThreadPoolExecutor | None = None,
):
  return piter(
      lambda x: map(fn, x),
      input_iterator,
      max_parallism=max_parallism,
      buffer_size=buffer_size,
      thread_pool=thread_pool,
  )


def is_stop_iteration(e: Exception) -> bool:
  return e is STOP_ITERATION or isinstance(e, StopIteration)


class PrefetchedIterator:
  """An iterator that can also prefetch before iterated."""

  def __init__(self, iterator, prefetch_size: int = 2):
    self._data = queue.SimpleQueue()
    self._returned = None
    iterator = iter(iterator)
    self._iterator = iterator
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
    logging.info('chainable: flush_prefetched: %s', len(result))
    return result

  def __next__(self):
    self.prefetch(1)
    if not self._data.empty():
      return self._data.get()
    else:
      logging.info('chainable: Generator exhausted from %s.', self._iterator)
      raise StopIteration(self._returned)

  def __iter__(self):
    return self

  def prefetch(self, num_items: int = 0):
    """Prefeches items from the undelrying generator."""
    while not self._exhausted and self._data.qsize() < (
        num_items or self._prefetch_size
    ):
      try:
        self._data.put(next(self._iterator))
        self._cnt += 1
        self._data_size += 1
      except StopIteration as e:
        self._exhausted = True
        self._returned = e.value
        logging.info(
            'chainable: prefetch exhausted after %d items.', self._cnt
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('chainable: Got error during prefetch.')
        if 'generator already executing' != str(e):
          self._exceptions.append(e)
          if len(self._exceptions) > 3:
            logging.exception('chainable: Too many errors, stop prefetching.')
            break

        time.sleep(0)


class _TeeIterator(Iterator[_ValueT]):
  """An iterator that record its inputs and can be re-iterated.

  This is a more restricted version to itertools.tee: it only be teeed once, and
  the main iterator has to be iterated first. But this does not suffer from
  out of memory issue as common.
  """

  def __init__(self, iterator: Iterable[_ValueT], *, buffer_size: int = 0):
    self._iterator = iter(iterator)
    self._buffer_size = buffer_size
    self._buffer = collections.deque(maxlen=buffer_size or None)
    self._exhausted = False
    self._returned = None

  def __next__(self):
    """Iterate through itself records every element it has seen by reference."""
    try:
      value = next(self._iterator)
    except StopIteration as e:
      self._exhausted = True
      self._returned = e.value
      raise e
    if self._buffer_size and len(self._buffer) == self._buffer_size:
      raise RuntimeError(
          f'Buffer reached capacity: {len(self._buffer)} / {self._buffer_size}.'
      )
    self._buffer.append(value)
    return value

  def __iter__(self):
    return self

  def tee(self) -> Iterator[_ValueT]:
    """Re-iterate the elements previously iterated in a FIFO order."""
    while True:
      try:
        yield self._buffer.popleft()
      except IndexError as e:
        if self._exhausted:
          return self._returned
        raise IndexError('No element left.') from e


def processed_with_inputs(
    process_fn: Callable[[Iterable[_InputT]], Iterable[_ValueT]],
    input_iterator: Iterable[_InputT],
    *,
    max_buffer_size: int = 0,
) -> Iterator[tuple[_InputT, _ValueT]]:
  """Zips the processed outputs with its inputs."""
  iter_input = _TeeIterator(input_iterator, buffer_size=max_buffer_size)
  iter_output = process_fn(iter_input)
  # Note that recital iterator has to be put after the input iterator so that
  # there are values to be recited.
  return zip(iter_output, iter_input.tee())


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
      for columns in zip(*map(sliced_by_batch_size, concated), strict=True):
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
