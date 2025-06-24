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
import bisect
import collections
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from concurrent import futures
import dataclasses as dc
import functools
import itertools as itt
import operator as op
import queue
import threading
from typing import Any, Generic, Protocol, Self, TypeVar

from absl import logging
from ml_metrics._src import types
from ml_metrics._src.utils import func_utils
import more_itertools as mit
import numpy as np


_ValueT = TypeVar('_ValueT')
_InputT = TypeVar('_InputT')
_MAX_BATCH_SIZE = 4096
_ITERATE_FN_MAX_THREADS = 256
_IGNORE_ERROR_TYPES = (ValueError, TypeError)
_RANDOM_ACCESS_BATCH_SIZE = 64


class _QueueLike(Protocol[_ValueT]):
  """Protocol of the same interfaces of queue.Queue."""

  def get_nowait(self) -> _ValueT:
    """Same as queue.Queue().get_nowait."""

  def put_nowait(self, value: _ValueT):
    """Same as queue.Queue().put_nowait."""

  def empty(self) -> bool:
    """Same as queue.Queue().empty."""


STOP_ITERATION = StopIteration()
# Only used in in process as indicator to skip, not meant to cross process.
_SKIP = '_SKIP'


def iter_ignore_error(it, error_return=None):
  """Yields the next element from an iterator, ignoring errors.

  Be careful when using this function, it can cause infinite loop if the
  underlying iterator cannot progress after an error, namely, calling `next()`
  on the iterator after an error can progress to the next element or raise
  `StopIteration`.

  Args:
    it: The iterator to ignore errors from.
    error_return: The value to return when an error is encountered.

  Yilds:
    The next element from the iterator, ignoring errors.
  """
  while True:
    try:
      yield next(it)
    except (StopIteration, StopAsyncIteration) as e:
      return e.value
    except _IGNORE_ERROR_TYPES:
      if error_return is not None:
        yield error_return
      continue


def map_ignore_error(
    fn: Callable[[_InputT], _ValueT], it: Iterable[_InputT]
) -> Iterator[_ValueT]:
  """Maps a function to an iterator, ignoring errors.

  Be careful when using this function, it can cause infinite loop if the
  underlying iterator cannot progress after an error, namely, calling `next()`
  on the iterator after an error can progress to the next element or raise
  `StopIteration`.

  Args:
    fn: The function to map.
    it: The iterator to ignore errors from.

  Returns:
    An iterator that yields that ignores errors.
  """
  return iter_ignore_error(map(fn, it))


class _IteratorWithLatest(Iterator[_ValueT]):
  """Iterator that returns the last item."""

  def __init__(self, it: Iterator[_ValueT]):
    self._it = it
    self._last_item = None

  def latest(self) -> _ValueT | None:
    return self._last_item

  def __next__(self) -> _ValueT:
    self._last_item = next(self._it)
    return self._last_item

  def __iter__(self):
    return self


def iter_with_latest(it: Iterable[_ValueT]) -> _IteratorWithLatest[_ValueT]:
  return _IteratorWithLatest(iter(it))


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


class _RangeIterator(Iterator[_ValueT]):
  """An range like iterator that iterate through random accessible data.

  The iterator can read ahead if the underlying data source supports random
  access with slicing. If failed, it will be decreased exponentially until to
  single element batch. Upon error even in single element batch, the iterator
  will skip the current index and raise the error. So that the next call to
  `__next__` will try to read the next index, making the error skippable.

  Attributes:
    data: The underlying data source.
    start: The start index of the iteration.
    stop: The stop index of the iteration.
    i: The current index of the iteration.
  """

  def __init__(
      self,
      data: types.RandomAccessible[_ValueT],
      start: int,
      stop: int | None,
      max_batch_size: int,
  ):
    self.data = data
    self.stop = len(data) if stop is None else stop
    self.start = start
    self.i = start
    self._batch_size = max_batch_size
    self._cache = collections.deque()

  def __next__(self):
    while not self._cache and self.i < self.stop:
      try:
        batch_size = self._batch_size
        if self._batch_size > 1:
          batch_size = min(self.i + self._batch_size, self.stop) - self.i
          self._cache.extend(self.data[self.i : self.i + batch_size])
        else:
          self._cache.append(self.data[self.i])
        self.i += batch_size
      except Exception as e:  # pylint: disable=broad-exception-caught
        # After falling back to single element batch, raise the exception and
        # skip the index by one.
        if self._batch_size == 1:
          self.i += self._batch_size
          raise
        # Fall back to smaller batch size when there is an error.
        self._batch_size = max(int(self._batch_size // 4), 1)
        logging.warning(
            'chainable: %s',
            f'Fall back to smaller {self._batch_size=} due to error: {e}.',
        )
    if self._cache:
      return self._cache.popleft()
    raise StopIteration()

  def __iter__(self):
    return self


@dc.dataclass(frozen=True, slots=True)
class _MergedSequenceIndex:
  seq_idx: int
  idx: int | None = None


# slice cannot be a type annotation, this is for documentation purpose.
_SliceT = Any


class MergedSequences(Generic[_ValueT]):
  """Merges multiple sequences into a single sequence."""

  def __init__(
      self,
      sequences: Iterable[types.RandomAccessible[_ValueT]],
      max_batch_size: int = 0,
  ):
    self._sequences = list(sequences)
    self._seq_idxs = [0]
    self._seq_idxs.extend(itt.accumulate(map(len, self._sequences), op.add))
    self._max_batch_size = max_batch_size or _RANDOM_ACCESS_BATCH_SIZE

  @property
  def sequences(self) -> list[types.RandomAccessible[_ValueT]]:
    return self._sequences

  def _index_slice(
      self,
      seq_idx: int,
      start: int = 0,
      stop: int | None = None,
  ) -> Iterator[_ValueT]:
    """Slices a sequence that supports random access."""
    return _RangeIterator(
        self._sequences[seq_idx], start, stop, self._max_batch_size
    )

  def __len__(self) -> int:
    return self._seq_idxs[-1]

  def _index(self, index: int | _SliceT) -> _MergedSequenceIndex:
    index = len(self) + index if index < 0 else index
    indices = self._seq_idxs
    idx_seq = bisect.bisect_left(indices, index)
    if idx_seq == len(indices) and index > indices[-1]:
      return _MergedSequenceIndex(idx_seq - 1)
    if index == indices[idx_seq]:
      return _MergedSequenceIndex(idx_seq, 0)
    return _MergedSequenceIndex(idx_seq - 1, index - indices[idx_seq - 1])

  def slice(self, slice_: _SliceT) -> Iterator[_ValueT]:
    """Slices the merged sequences."""
    if slice_.step is not None:
      raise NotImplementedError(f'step is not supported, got {slice_}')
    start = self._index(slice_.start or 0)
    stop = self._index(len(self) if slice_.stop is None else slice_.stop)
    if start.seq_idx == len(self._sequences):
      return iter(())
    if start.seq_idx == stop.seq_idx:
      return self._index_slice(start.seq_idx, start.idx, stop.idx)
    # Chain multiple sequences together with correct slices.
    sequences = [self._index_slice(start.seq_idx, start.idx, None)]
    for i_seq in range(start.seq_idx + 1, stop.seq_idx):
      sequences.append(self._index_slice(i_seq))
    if stop.idx:
      sequences.append(self._index_slice(stop.seq_idx, 0, stop.idx))
    return itt.chain.from_iterable(sequences)

  def __getitem__(self, index: int | Any) -> _ValueT | Iterator[_ValueT]:
    if isinstance(index, slice):
      return self.slice(index)
    multi_idx = self._index(index)
    try:
      return self._sequences[multi_idx.seq_idx][multi_idx.idx]
    except IndexError:
      raise IndexError(f'Index {index} is out of range.') from None

  def __iter__(self):
    return self.slice(slice(None))


class MultiplexIterator(Iterator[_ValueT], types.Stoppable, types.Recoverable):
  """An iterator that merges multiple iterables."""

  def __init__(
      self,
      *,
      data_sources: Sequence[Iterable[Any]],
      iter_fn: Callable[[Iterable[Any]], Iterator[_ValueT]] | None = None,
      parallism: int = 0,
      name: str = '',
  ):
    self._name = name
    self._parallism = parallism
    self._data_sources = list(data_sources)
    self._iter_fn = iter_fn
    self._source_iterators = [iter(ds) for ds in data_sources]
    self._thread_pool = None
    input_iterator = None
    if not parallism:
      if len(self._source_iterators) == 1:
        input_iterator = self._source_iterators[0]
      elif self._source_iterators:
        input_iterator = itt.chain(*self._source_iterators)
      self._iterator = input_iterator
      if iter_fn is not None:
        self._iterator = _get_iterate_fn(iter_fn, self._iterator)
      return

    self._thread_pool = futures.ThreadPoolExecutor(
        max_workers=parallism,
        thread_name_prefix=f'multiplex_pool: "{self.name}"',
    )
    if len(self._source_iterators) > 1:
      iterators = self._source_iterators
      if iter_fn is not None:
        iterators = [iter_fn(ds) for ds in iterators]
      iter_q = piter_multiplex(
          input_iterators=iterators,
          thread_pool=self._thread_pool,
          buffer_size=parallism * 3,
      )
      self._iterator = iter(iter_q)
    else:
      if self._source_iterators:
        input_iterator = self._source_iterators[0]
      iter_q = piter_fn(
          iter_fn,
          input_iterable=input_iterator,
          thread_pool=self._thread_pool,
          parallism=parallism,
          buffer_size=parallism * 3,
      )
      self._iterator = iter(iter_q)

  @property
  def name(self) -> str:
    return self._name

  def maybe_stop(self):
    if isinstance(self._iterator, types.Stoppable):
      self._iterator.maybe_stop()
    if self._thread_pool is not None:
      self._thread_pool.shutdown()

  def from_state(self, states: Sequence[Any], **kwargs) -> Self:
    data_sources = []
    for data_source, ds_state in zip(self._data_sources, states, strict=True):
      if not types.is_recoverable(data_source):
        raise TypeError(f'Data source is not recoverable, got {data_source=}.')
      data_sources.append(data_source.from_state(ds_state))
    return self.__class__(data_sources=data_sources, **kwargs)

  @property
  def state(self) -> list[Any]:
    states = []
    for iterator in self._source_iterators:
      if not types.is_recoverable(iterator):
        raise TypeError(f'Data source is not serializable, got {iterator=}.')
      states.append(iterator.state)
    return states

  def __next__(self):
    try:
      return next(self._iterator)
    except StopIteration:
      self.maybe_stop()
      raise
    except KeyboardInterrupt:
      self.maybe_stop()
      raise
    except Exception:
      logging.exception('chainable: %s', f'error iterating "{self.name}".')
      self.maybe_stop()
      raise

  def __iter__(self):
    return self


@dc.dataclass(slots=True, eq=False)
class Progress:
  cnt: int = 0


class IterableQueue(Generic[_ValueT], abc.ABC):
  """Base implementation and interfaces for an (Async)Iterables queue."""

  @abc.abstractmethod
  def get(self) -> _ValueT:
    """Dequeue an element from the queue."""

  @abc.abstractmethod
  def get_batch(self) -> list[_ValueT]:
    """Dequeue a batch of elements from the queue."""

  def dequeue_as_iterator(self, num_steps: int = -1) -> Iterator[_ValueT]:
    return DequeueIterator(self, num_steps=num_steps)

  def __iter__(self):
    return self.dequeue_as_iterator()


class AsyncIterableQueue(IterableQueue[_ValueT]):
  """Base implementation and interfaces for an (Async)Iterables queue."""

  @abc.abstractmethod
  async def async_get(self):
    """Gets an element from the queue."""

  @abc.abstractmethod
  async def async_get_batch(self):
    """Gets a batch of elements from the queue."""

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


def _async_get_batch(iterator_queue: IterableQueue):
  try:
    return iterator_queue.get_batch()
  except StopIteration as e:
    raise StopAsyncIteration(*e.args) from e


class IteratorQueue(IterableQueue[_ValueT]):
  """Enqueue elements from an iterator and record exhausted and returned.

  Attributes:
    name: The name of the queue.
    timeout: The timeout in seconds for enqueue and dequeue.
    ignore_error: Whether to ignore errors during enqueue and dequeue.
    exhausted: Whether the queue is exhausted.
    exception: The exception raised during enqueue.
    enqueue_done: Whether all the enqueue threads exhausted their input
      iterators.
    returned: The values returned from the input generators.
    progress: The progress of the queue.
  """

  timeout: float | None
  name: str
  ignore_error: bool

  def __init__(
      self,
      queue_or_size: int | _QueueLike[_ValueT] = 0,
      *,
      name: str = '',
      timeout: float | None = None,
      ignore_error: bool = False,
      max_batch_size: int = 0,
      max_enqueuer: int = 0,
  ):
    if isinstance(queue_or_size, int):
      self._queue = self._default_queue(queue_or_size)
    else:
      self._queue = queue_or_size
    self._max_batch_size = max_batch_size or _MAX_BATCH_SIZE
    self.name = name
    self.timeout = timeout
    # The events here manages enqueue (put) and dequeue (get).
    self._not_empty, self._not_full = threading.Event(), threading.Event()
    # The lock here protects the mutation and access to all states.
    self._states_lock = threading.RLock()
    self._progress = Progress()
    self._returned = []
    self._exception = None
    self._exhausted = False
    self._max_enqueuer = max_enqueuer
    self._enqueue_start = 0
    self._enqueue_stop = 0
    self.ignore_error = ignore_error

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

  def _set_exhausted(self):
    self._exhausted = True
    self._not_empty.set()
    remain = self._enqueue_start - self._enqueue_stop
    has_exception = self.exception is not None
    logging.debug(
        'chainable: %s', f'"{self.name}" exhausted {has_exception=}, {remain=}'
    )

  def __bool__(self):
    return not self.exhausted

  @property
  def exception(self) -> Exception | None:
    return self._exception

  @property
  def enqueue_done(self) -> bool:
    """Indicates whether there is ongoing enqueuer."""
    if self._exception:
      return True
    # If max_enqueuer is not set, it means the no enqueuer has started yet.
    if not self._max_enqueuer:
      return False
    return self._enqueue_start == self._enqueue_stop == self._max_enqueuer

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
      self._get_nowait_impl()
    finally:
      self._states_lock.release()

  def _get_nowait_impl(self) -> _ValueT:
    """Gets an element from the queue, raises Empty immediately if empty.

    Does not acquire the lock to allow for callers that repeatedly call
    get_nowait.
    
    Returns:
      The result list.
    """
    try:
      result = self._queue.get_nowait()
      # Premeptively check if the queue is exhausted to avoid a second call.
      if self._queue.empty() and self.enqueue_done:
        self._set_exhausted()
      return result
    except (queue.Empty, asyncio.QueueEmpty) as e:
      # No need to rasie from since these are the actual error.
      if self._exhausted:
        raise self.exception or StopIteration(*self.returned)
      if self.enqueue_done:
        self._set_exhausted()
        raise self.exception or StopIteration(*self.returned)
      self._not_empty.clear()
      raise e
    except StopIteration as e:
      raise e
    except Exception as e:  # pylint: disable=broad-exception-caught
      e.add_note(f'Exception during Dequeueing "{self.name}".')
      logging.exception('chainable: %s', f'"{self.name}" Dequeue failed.')
      raise e

  def _build_results_noblock(self, max_batch_size: int = 0) -> list[_ValueT]:
    """Gets elements from the queue, and returns if any items are available."""
    result = []
    #  Dequeue all elements in the queue if max_batch_size is not set.
    self._states_lock.acquire()
    try:
      while not max_batch_size or len(result) < max_batch_size:
        try:
          result.append(self.get_nowait())
        except (queue.Empty, asyncio.QueueEmpty):
          break
        except Exception as e:  # pylint: disable=broad-exception-caught
          exhausted = is_stop_iteration(e)
          if (exhausted and result) or (not exhausted and self.ignore_error):
            break
          raise e
    finally:
      self._states_lock.release()
    return result

  def _build_results_block(
      self, max_batch_size: int = 0, min_batch_size: int = 0
  ) -> list[_ValueT]:
    """Gets elements from the queue, waits for timeout if empty."""
    max_batch_size = max_batch_size or self._max_batch_size
    result = []
    #  Dequeue all elements in the queue if max_batch_size is not set.
    while not max_batch_size or len(result) < max_batch_size:
      try:
        result.append(self.get_nowait())
      except (queue.Empty, asyncio.QueueEmpty) as e:
        if len(result) >= min_batch_size:
          break

        if result:
          self._not_full.set()
        logging.debug(
            'chainable: %s',
            f'"{self.name}" Dequeue empty, got {len(result)} elements, waiting'
            f' {self.enqueue_done=} {self._not_empty.is_set()=}.',
        )
        # this could cause a deadlock if min_batch_size is set
        # let's do a noblock while loop implementation that only locks once.
        if self._not_empty.wait(timeout=self.timeout):
          continue

        raise TimeoutError(
            f'"{self.name}" dequeue timeout after {self.timeout}secs.'
        ) from e
      except Exception as e:  # pylint: disable=broad-exception-caught
        exhausted = is_stop_iteration(e)
        if (exhausted and result) or (not exhausted and self.ignore_error):
          break
        raise e
    return result

  def get_batch(
      self, max_batch_size: int = 0, *, min_batch_size: int = 1
  ) -> list[_ValueT]:
    """Gets elements from the queue, waits for timeout if empty.

    Args:
      max_batch_size: The number of elements to be dequeued. If 0, it will wait
        for at least one element. If set, it will wait at most max_batch_size
        elements.
      min_batch_size: The minimum number of elements to be dequeued. Default to
        1 so that get_batch() default block on getting one element.

    Returns:
      A list of dequeued elements.
    """
    max_batch_size = max_batch_size or self._max_batch_size
    #  Dequeue all elements in the queue if max_batch_size is not set.
    if min_batch_size > 1:
      result = self._build_results_block(max_batch_size, min_batch_size)
    else:
      result = self._build_results_noblock(max_batch_size)

    self._not_full.set()
    logging.debug(
        'chainable: %s', f'"{self.name}" Dequeued {len(result)} elements.'
    )
    return result

  def get(self) -> _ValueT:
    """Gets an element from the queue, waits for timeout if empty."""
    return self.get_batch(1, min_batch_size=1)

  def put_nowait(self, value: _ValueT = _SKIP, *, stop_iteration=None) -> None:
    """Puts a value to the queue, raises if queue is full immediately."""
    self._states_lock.acquire()
    try:
      if value is not _SKIP:
        self._queue.put_nowait(value)
      if is_stop_iteration(stop_iteration):
        self._stop_enqueue(*stop_iteration.args)
      elif stop_iteration:
        self._exception = stop_iteration
        self._stop_enqueue()
      self._not_empty.set()
      self._progress.cnt += 1
      logging.debug(
          'chainable: %s', f'"{self.name}" enqueued cnt {self._progress.cnt}.'
      )
    except (queue.Full, asyncio.QueueFull):
      self._not_full.clear()
      raise
    finally:
      self._states_lock.release()

  def put(self, value: _ValueT = _SKIP, *, stop_iteration=None) -> None:
    """Puts a value to the queue and optionally stop enqueue in one transaction."""
    while not self.enqueue_done:
      try:
        return self.put_nowait(value, stop_iteration=stop_iteration)
      except (queue.Full, asyncio.QueueFull) as e:
        logging.debug('chainable: %s', f'"{self.name}" enqueue full, waiting')
        if self._not_full.wait(timeout=self.timeout):
          continue
        raise TimeoutError(f'Enqueue timeout={self.timeout}secs.') from e

  def _start_enqueue(self):
    with self._states_lock:
      self._enqueue_start += 1
      self._max_enqueuer = max(self._max_enqueuer, self._enqueue_start)

  def _stop_enqueue(self, *values):
    """Stops enqueueing and records the returned values."""
    with self._states_lock:
      self._enqueue_stop += 1
      self._enqueue_stop = min(self._enqueue_stop, self._enqueue_start)
      self._returned.extend(values)
      if self.enqueue_done:
        self._not_empty.set()
        logging.debug('chainable: %s', f'"{self.name}" enqueue done, notify.')
    logging.debug(
        'chainable: %s',
        f'"{self.name}" enqueue stop, remaining='
        f'{self._enqueue_start - self._enqueue_stop}/{self._max_enqueuer}, '
        f'cnt:{self._progress.cnt}, exception={self.exception is not None}',
    )

  def maybe_stop(self, exc: Exception | None = None):
    """Stops the producer and optionally terminates the consumer.

    When non-StopIteration exception is provided, this will also mark itself
    exhausted so that any consumer will be terminated immediately.

    Args:
      exc: The exception to be raised by the consumer, the queue will be
        teriminated immediately if this is not StopIteration.

    Returns:
      None.
    """
    exc = exc or StopIteration()
    with self._states_lock:
      self._enqueue_stop = self._enqueue_start = self._max_enqueuer
      if not is_stop_iteration(exc):
        self._exception = exc
      assert self.enqueue_done, f'{self._enqueue_stop=}, {self._enqueue_start=}'
      self._not_full.set()
      if not is_stop_iteration(exc):
        # Immediately terminate the consumer side for enqueue error.
        self._set_exhausted()
      else:
        self._not_empty.set()
    logging.info('chainable: %s', f'"{self.name}" stopping enqueue.')

  def enqueue_from_iterator(self, iterator: Iterable[_ValueT]) -> None:
    """Iterates through a generator while enqueue its elements."""
    iterator = iter(iterator)
    self._start_enqueue()
    buffer = []
    while not self.enqueue_done:
      try:
        buffer.append(next(iterator))
        # Skip the first value, so only enqueue with one element delay.
        if len(buffer) == 2:
          self.put(buffer.pop(0))
      except StopIteration as e:
        value = buffer.pop() if buffer else _SKIP
        self.put(value, stop_iteration=e)
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Need to go pass this exception assuming the iterator can skip error.
        if self.ignore_error:
          logging.exception(
              'chainable: %s',
              f'"{self.name}" enqueue error ignored, stacktrace:',
          )
          continue
        e.add_note(f'Exception during enqueueing "{self.name}".')
        value = buffer.pop() if buffer else _SKIP
        self.put(value, stop_iteration=e)
        logging.exception('chainable: %s', f'"{self.name}" enqueue failed.')
        raise e


class _ThreadSafeIterator(Iterator[_ValueT]):
  """Converts an iterator to be thread safe."""

  def __init__(self, iterable: Iterable[_ValueT]):
    self._iterator = iter(iterable)
    self._lock = threading.Lock()

  def __next__(self):
    with self._lock:
      return next(self._iterator)

  def __iter__(self):
    return self


class DequeueIterator(Iterator[_ValueT], types.Stoppable):
  """An iterator that dequeues from an IteratorQueue."""

  def __init__(self, q: IterableQueue[_ValueT], *, num_steps: int = -1):
    self._iterator_queue = q
    self._num_steps = num_steps
    self._cnt = 0
    self._run_until_exhausted = num_steps < 0
    self._cache = collections.deque()

  def maybe_stop(self):
    assert isinstance(self._iterator_queue, IteratorQueue)
    self._iterator_queue.maybe_stop()

  def __next__(self) -> _ValueT:
    if not self._run_until_exhausted and self._cnt == self._num_steps:
      self.maybe_stop()
      raise StopIteration()
    if not self._cache:
      self._cache.extend(self._iterator_queue.get_batch())
    self._cnt += 1
    return self._cache.popleft()

  def __iter__(self):
    return self


_MaybeAwaitable = _ValueT | Awaitable[_ValueT]


class AsyncIteratorQueue(IteratorQueue[_ValueT], AsyncIterableQueue[_ValueT]):
  """A queue that can enqueue from and dequeue to an iterator.

  Attributes:
    name: The name of the queue.
    timeout: The timeout in seconds for enqueue and dequeue.
    thread_pool: The thread pool to be used for enqueue and dequeue.
    ignore_error: Whether to ignore errors during enqueue and dequeue.
  """
  _thread_pool: futures.ThreadPoolExecutor | None

  def __init__(
      self,
      queue_or_size: int | _QueueLike[_ValueT] = 0,
      *,
      name: str = '',
      timeout: float | None = None,
      thread_pool: futures.ThreadPoolExecutor | None = None,
      ignore_error: bool = False,
      max_batch_size: int = _MAX_BATCH_SIZE,
  ):
    super().__init__(
        queue_or_size=queue_or_size,
        name=name,
        timeout=timeout,
        ignore_error=ignore_error,
        max_batch_size=max_batch_size,
    )
    self._thread_pool = thread_pool

  @classmethod
  def _default_queue(cls, maxsize: int):
    return asyncio.Queue(maxsize=maxsize)

  async def async_get_batch(self):
    """Gets a batch of elements from the queue."""
    logging.debug('chainable: %s', f'{self.name} async dequeueing')
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        self._thread_pool, _async_get_batch, self
    )
    logging.debug(
        'chainable: %s', f'"{self.name}" async dequeued {len(result)}'
    )
    return result

  async def async_get(self):
    """Gets an element from the queue."""
    logging.debug('chainable: %s', f'{self.name} async dequeueing')
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self._thread_pool, _async_get, self)

  async def async_put(self, value: _ValueT):
    logging.debug(
        'chainable: %s', f'{self.name} async enqueueing a {type(value)}'
    )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self._thread_pool, self.put, value)

  async def async_enqueue_from_iterator(
      self, iterator: _MaybeAwaitable[AsyncIterable[_ValueT]]
  ):
    """Iterates through a generator while enqueue its elements."""
    if isinstance(iterator, Awaitable):
      iterator = await iterator
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
        logging.exception('chainable: %s', f'{self.name} enqueue failed.')
        self._exception = e
        self._stop_enqueue()
        if self.ignore_error:
          return
        raise e


class _AsyncDequeueIterator:
  """An async iterator that dequeues from an AsyncIteratorQueue."""

  def __init__(self, q: AsyncIterableQueue, *, num_steps: int = -1):
    self._iterator_queue = q
    self._num_steps = num_steps
    self._cnt = 0
    self._run_until_exhausted = num_steps < 0
    self._cache = collections.deque()

  async def __anext__(self):
    if self._run_until_exhausted or self._cnt < self._num_steps:
      # AsyncIteratorQueue.get() can raise StopAsyncIteration.
      name = getattr(self._iterator_queue, 'name', '')
      try:
        if not self._cache:
          self._cache.extend(await self._iterator_queue.async_get_batch())
      except StopAsyncIteration as e:
        logging.info(
            'chainable: %s',
            f'async iterator "{name}" exhausted after {self._cnt} batches'
            f' with {len(e.args)} returns.',
        )
        raise e
      self._cnt += 1
      logging.debug(
          'chainable: %s', f'"{name}" async iter yield batch cnt: {self._cnt}'
      )
      return self._cache.popleft()
    else:
      raise StopAsyncIteration()

  def __aiter__(self):
    return self


def _get_thread_pool(
    thread_pool: futures.ThreadPoolExecutor | None = None,
) -> futures.ThreadPoolExecutor:
  if thread_pool is None:
    thread_pool = futures.ThreadPoolExecutor(thread_name_prefix='piter')
  return thread_pool


def _get_iterate_fn(
    iter_fn: Callable[..., Iterable[_ValueT]] | None,
    inputs: Iterable[_ValueT] | None,
) -> Iterable[_ValueT]:
  assert iter_fn or inputs is not None, f'{iter_fn=}, {inputs=}'
  iter_fn = iter_fn or iter
  return iter_fn(inputs) if inputs is not None else iter_fn()


def piter_multiplex(
    input_iterators: Iterable[Iterable[_ValueT]],
    thread_pool: futures.ThreadPoolExecutor,
    buffer_size: int = 0,
    max_batch_size: int = 0,
) -> Iterable[_ValueT]:
  """Call a chain of functions in sequence concurrently with multithreads.

  Args:
    input_iterators: The input iterators to be passed to the iterator_fn. More
      than one input iterators are supported and will be iterated concurrently.
    thread_pool: The thread pool to be used.
    buffer_size: The buffer size of the queue.
    max_batch_size: The max batch size when dequeuing.

  Returns:
    An iterable that iterates through the chain of functions.
  """
  input_iterators = tuple(input_iterators)
  if not input_iterators:
    raise ValueError('input_iterators has to be provided.')

  result_queue = IteratorQueue(
      buffer_size,
      max_batch_size=max_batch_size,
      name='piter_multiplex_q',
      max_enqueuer=len(input_iterators),
  )
  thread_pool = _get_thread_pool(thread_pool)
  for iterator in input_iterators:
    thread_pool.submit(result_queue.enqueue_from_iterator, iterator)
  return result_queue


def piter_fn(
    iter_fn: Callable[..., Iterable[_ValueT]] | None = None,
    *,
    input_iterable: Iterable[Any] | None = None,
    thread_pool: futures.ThreadPoolExecutor | None,
    parallism: int = 1,
    buffer_size: int = 0,
) -> Iterable[_ValueT]:
  """Call a chain of functions in sequence concurrently with multithreads.

  Args:
    iter_fn: A generator function that takes an iterable as input.
    input_iterable: The input iterators to be passed to the iter_fn. More
    thread_pool: The thread pool to be used. than one input iterators are
      supported and will be iterated concurrently.
    parallism: The maximum number of threads to be used for iter_fn. If 0, the
      iter_fn will be called in process; If >= 1, the iter_fn is called
      concurrently.
    buffer_size: The buffer size of the queue.

  Returns:
    An iterable that iterates through the chain of functions.
  """
  if not parallism:
    return _get_iterate_fn(iter_fn, input_iterable)

  if thread_pool is None:
    raise ValueError('thread_pool required, got None.')
  if input_iterable is not None:
    input_iterable = _ThreadSafeIterator(input_iterable)
  its = [_get_iterate_fn(iter_fn, input_iterable) for _ in range(parallism)]
  return piter_multiplex(its, thread_pool, buffer_size)


def piter(
    iterator_fn: Callable[..., Iterable[_ValueT]] | None = None,
    *,
    input_iterators: Iterable[Iterable[Any]] = (),
    max_parallism: int = 1,
    buffer_size: int = 0,
    thread_pool: futures.ThreadPoolExecutor | None = None,
) -> Iterable[_ValueT]:
  """Call a chain of functions in sequence concurrently with multithreads.

  Args:
    iterator_fn: A generator function that takes an iterable as input.
    input_iterators: The input iterators to be passed to the iterator_fn. More
      than one input iterators are supported and will be iterated concurrently.
    max_parallism: The maximum number of threads to be used for iterator_fn. If
      0, the iterator_fn will be called in process; If >= 1, the iterator_fn is
      called concurrently with the specified number of threads including with
      only one thread.
    buffer_size: The buffer size of the queue.
    thread_pool: The thread pool to be used. If None, a new thread pool will be
      created.

  Returns:
    An iterable that iterates through the chain of functions.
  """
  input_iterators = tuple(input_iterators)
  if iterator_fn is None and not input_iterators:
    raise ValueError('iterator_fn or input_iterators has to be provided.')
  input_iterable = None
  # No parallelism at all, use the input iterator directly.
  if len(input_iterators) == 1:
    input_iterable = input_iterators[0]
  elif input_iterators:
    # The input_queue uses a max_batch_size of 1 to ensure that the output_queue
    # is not consuming too many elements that leads to imbalanced parallelism.
    max_batch_size = 1 if max_parallism > 1 and iterator_fn is not None else 0
    thread_pool = _get_thread_pool(thread_pool)
    input_iterable = piter_multiplex(
        input_iterators,
        thread_pool=thread_pool,
        # Limit buffer size to avoid OOM.
        buffer_size=buffer_size or max_parallism,
        max_batch_size=max_batch_size,
    )
  if iterator_fn is None:
    assert input_iterable is not None
    return input_iterable
  thread_pool = _get_thread_pool(thread_pool)
  return piter_fn(
      iterator_fn,
      thread_pool=thread_pool,
      input_iterable=input_iterable,
      parallism=max_parallism,
      buffer_size=buffer_size,
  )


def pmap(
    fn: Callable[..., Any],
    input_iterator: Iterable[Any] = (),
    *,
    max_parallism: int = 1,
    buffer_size: int = 0,
    thread_pool: futures.ThreadPoolExecutor | None = None,
):
  """A wrapper of piter to make it easier to use with pmap."""
  if not max_parallism:
    return map(fn, input_iterator)
  if thread_pool is None:
    thread_pool = futures.ThreadPoolExecutor(max_workers=1 + max_parallism)
  return piter_fn(
      lambda it: map(fn, it),
      thread_pool=thread_pool,
      input_iterable=input_iterator,
      parallism=max_parallism,
      buffer_size=buffer_size,
  )


def iterate_fn(
    fn=None,
    *,
    multithread: bool = False,
) -> Callable[..., tuple[_ValueT, ...] | _ValueT]:
  """Wraps a callable that transposes the input and the output.

  This is to transpose column-oriented data to row-oriented data before
  passing it to the wrapped function, and transpose the output back to
  column-oriented data.

  Args:
    fn: the function to consume and output row-oriented data.
    multithread: Whether to use multithreading.

  Returns:
    A function that consumes the column oriented inputs.
  """

  def decorator(fn):
    thread_pool = None
    if multithread:
      thread_pool = func_utils.SingletonThreadPool(
          _ITERATE_FN_MAX_THREADS, thread_name_prefix='iterate_fn'
      )

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs) -> tuple[_ValueT, ...] | _ValueT:
      num_args = len(args)
      kwargs_keys = tuple(kwargs.keys())
      # Unpack the args and kwargs's values first.
      args_values = mit.zip_broadcast(*args, *kwargs.values())
      # Recompose the kwargs keys back.
      args_and_kwargs = (
          (values[:num_args], dict(zip(kwargs_keys, values[num_args:])))
          for values in args_values
      )
      if not thread_pool:
        outputs = list(
            fn(*row_inputs, **row_kwinputs)
            for row_inputs, row_kwinputs in args_and_kwargs
        )
      else:
        states = []
        for row_inputs, row_kwinputs in args_and_kwargs:
          states.append(thread_pool.submit(fn, *row_inputs, **row_kwinputs))
        outputs = list(x.result() for x in futures.as_completed(states))
      # Only transpose when fn returns multiple items (exact tuple).
      if outputs and type(outputs[0]) is tuple:  # pylint: disable=unidiomatic-typecheck
        return tuple(zip(*outputs))
      return outputs

    wrapped_fn.thread_pool = thread_pool
    return wrapped_fn

  return decorator(fn) if fn is not None else decorator


def is_stop_iteration(e: Exception) -> bool:
  return e is STOP_ITERATION or isinstance(e, StopIteration)


class _TeeIterator(Iterator[_ValueT]):
  """An iterator that record its inputs and can be re-iterated.

  This is a more restricted version to itertools.tee: it only be teeed once, and
  the main iterator has to be iterated first. But this does not suffer from
  out of memory issue as common.
  """

  def __init__(self, iterator: Iterator[_ValueT], *, buffer_size: int = 0):
    self._iterator = iterator
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
    process_fn: Callable[[Iterator[_InputT]], Iterator[_ValueT]],
    input_iterator: Iterator[_InputT],
    *,
    max_buffer_size: int = 0,
    ignore_error: bool = False,
) -> Iterator[tuple[_ValueT, _InputT]]:
  """Zips the processed outputs with its inputs."""
  iter_input = _TeeIterator(input_iterator, buffer_size=max_buffer_size)
  iter_output = process_fn(iter_input)
  if ignore_error:
    iter_output = iter_ignore_error(iter_output, error_return=_SKIP)
    # Note that recital iterator has to be put after the input iterator so that
    # there are values to be recited.
    return (
        (output, input)
        for output, input in zip(iter_output, iter_input.tee())
        if output is not _SKIP
    )
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


def _pad(data: Any, pad: Any, batch_size: int):
  if hasattr(data, '__array__'):
    return np.pad(data, (0, batch_size - data.shape[0]), constant_values=pad)
  elif isinstance(data, list):
    return list(mit.padded(data, pad, batch_size))
  elif isinstance(data, tuple):
    return tuple(mit.padded(data, pad, batch_size))
  else:
    raise TypeError(
        f'Unsupported container type: {type(data)}, only list, tuple and numpy'
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


def rebatched_args(
    tuples: Iterator[tuple[_ValueT, ...]],
    batch_size: int = 0,
    *,
    num_columns: int = 0,
    pad: Any = None,
) -> Iterator[tuple[_ValueT, ...]]:
  """Merges and concatenates n batches of tuples while iterating."""
  if not batch_size:
    yield from tuples
    return

  if not num_columns:
    first_batch = mit.first(tuples)
    tuples = mit.prepend(first_batch, tuples)
    num_columns = len(first_batch)
    logging.debug('chainable: %s', f'rebatched_tuples: {num_columns=}')
  column_buffer = [[] for _ in range(num_columns)]
  batch_sizes = np.zeros(num_columns, dtype=int)
  exhausted = False
  last_columns = None
  while not exhausted:
    if (batch := next(tuples, None)) is None:
      exhausted = True
    else:
      if len(batch) != num_columns:
        raise ValueError(f'Mismatched columns: {len(batch)} != {num_columns=}')
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
      if _batch_size(last_columns[0]) == batch_size:
        yield last_columns
      elif exhausted and pad is not None:
        yield tuple(_pad(col, pad, batch_size) for col in last_columns)
      elif exhausted:
        yield last_columns
      else:
        column_buffer = [[column] for column in last_columns]
        batch_sizes += [_batch_size(column) for column in last_columns]
      last_columns = None
