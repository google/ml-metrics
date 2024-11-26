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
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from concurrent import futures
import dataclasses as dc
import functools
import itertools as itt
import queue
import threading
from typing import Any, Generic, Protocol, Self, TypeVar

from absl import logging
from ml_metrics._src.utils import func_utils
import more_itertools as mit
import numpy as np


_ValueT = TypeVar('_ValueT')
_InputT = TypeVar('_InputT')
_MAX_BATCH_SIZE = 4096
_ITERATE_FN_MAX_THREADS = 256


class _QueueLike(Protocol[_ValueT]):
  """Protocol of the same interfaces of queue.Queue."""

  def get_nowait(self) -> _ValueT:
    """Same as queue.Queue().get_nowait."""

  def put_nowait(self, value: _ValueT):
    """Same as queue.Queue().put_nowait."""

  def empty(self) -> bool:
    """Same as queue.Queue().empty."""


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

  @abc.abstractmethod
  def get_batch(self) -> list[_ValueT]:
    """Dequeue a batch of elements from the queue."""

  def dequeue_as_iterator(self, num_steps: int = -1) -> Iterator[_ValueT]:
    return _DequeueIterator(self, num_steps=num_steps)

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


def _release_and_notify(
    lock: threading.Lock | threading.Condition | threading.RLock,
    notify: threading.Condition,
    notify_all: bool = False,
):
  """Temporarily releases the lock and notifies the notify_lock."""
  lock.release()
  try:
    with notify:
      if notify_all:
        notify.notify_all()
      else:
        notify.notify()
  finally:
    lock.acquire()


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
  _queue: _QueueLike[_ValueT]
  _max_batch_size: int
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
  _run_enqueue: bool

  def __init__(
      self,
      queue_or_size: int | _QueueLike[_ValueT] = 0,
      *,
      name: str = '',
      timeout: float | None = None,
      ignore_error: bool = False,
      max_batch_size: int = 0,
  ):
    if isinstance(queue_or_size, int):
      self._queue = self._default_queue(queue_or_size)
    else:
      self._queue = queue_or_size
    self._max_batch_size = max_batch_size or _MAX_BATCH_SIZE
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
    self._run_enqueue = True
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
    logging.debug(
        'chainable: "%s" dequeue exhausted with_exception=%s, remaining %d',
        self.name,
        self.exception is not None,
        self._iterator_cnt,
    )
    self._exhausted = True
    self._dequeue_lock.notify_all()

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
      result = self._queue.get_nowait()
      # Premeptively check if the queue is exhausted to avoid a second call.
      if self._queue.empty() and self.enqueue_done:
        self._set_exhausted()
      return result
    except (queue.Empty, asyncio.QueueEmpty) as e:
      if self._exhausted:
        raise self.exception or StopIteration(*self.returned) from e
      if self.enqueue_done:
        self._set_exhausted()
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

  def get_batch(
      self, max_batch_size: int = 0, *, block: bool = False
  ) -> list[_ValueT]:
    """Gets elements from the queue, waits for timeout if empty.

    Args:
      max_batch_size: The number of elements to be dequeued. If 0, it will wait
        for at least one element. If set, it will wait at most max_batch_size
        elements.
      block: Whether to block until there are enough elements before returning.

    Returns:
      A list of dequeued elements.
    """
    max_batch_size = max_batch_size or self._max_batch_size
    result = []
    with self._dequeue_lock:
      while not max_batch_size or len(result) < max_batch_size:
        try:
          result.append(self.get_nowait())
        except (queue.Empty, asyncio.QueueEmpty) as e:
          if (not block and result) or (
              block and max_batch_size and len(result) == max_batch_size
          ):
            break
          if result:
            _release_and_notify(self._dequeue_lock, notify=self._enqueue_lock)
          logging.debug('chainable: "%s" dequeue empty, waiting', self.name)
          if self._dequeue_lock.wait(timeout=self.timeout):
            continue
          raise TimeoutError(
              f'"{self.name}" dequeue timeout={self.timeout}secs.'
          ) from e
        except Exception as e:  # pylint: disable=broad-exception-caught
          exhausted = is_stop_iteration(e)
          if (exhausted and result) or (not exhausted and self.ignore_error):
            break
          raise e
    with self._enqueue_lock:
      self._enqueue_lock.notify()
    logging.debug(
        'chainable: %s', f'"{self.name}" dequeued {len(result)} batches'
    )
    return result

  def get(self) -> _ValueT:
    """Gets an element from the queue, waits for timeout if empty."""
    with self._dequeue_lock:
      while True:
        try:
          value = self.get_nowait()
          _release_and_notify(self._dequeue_lock, notify=self._enqueue_lock)
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
      while self._run_enqueue:
        try:
          self.put_nowait(value)
          _release_and_notify(self._enqueue_lock, notify=self._dequeue_lock)
          return
        except (queue.Full, asyncio.QueueFull) as e:
          logging.debug('chainable: "%s" enqueue full, waiting', self.name)
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
        _release_and_notify(
            self._states_lock, notify=self._dequeue_lock, notify_all=True
        )
        logging.debug('chainable: "%s" enqueue done, notify all', self.name)

  def stop_enqueue(self):
    self._run_enqueue = False
    with self._states_lock:
      self._iterator_cnt = 0
    with self._enqueue_lock:
      self._enqueue_lock.notify_all()
    with self._dequeue_lock:
      self._dequeue_lock.notify_all()

  def enqueue_from_iterator(self, iterator: Iterable[_ValueT]):
    """Iterates through a generator while enqueue its elements."""
    iterator = iter(iterator)
    self._start_enqueue()
    self._run_enqueue = True
    while self._run_enqueue:
      try:
        self.put(next(iterator))
      except StopIteration as e:
        self._stop_enqueue(*e.args)
        return
      except Exception as e:  # pylint: disable=broad-exception-caught
        e.add_note(f'Exception during enqueueing "{self.name}".')
        logging.exception('chainable: "%s" enqueue failed.', self.name)
        self._exception = e
        self._stop_enqueue()
        if self.ignore_error:
          return
        raise e


class _DequeueIterator:
  """An iterator that dequeues from an IteratorQueue."""

  def __init__(self, q: IterableQueue, *, num_steps: int = -1):
    self._iterator_queue = q
    self._num_steps = num_steps
    self._cnt = 0
    self._run_until_exhausted = num_steps < 0
    self._cache = collections.deque()

  def __next__(self):
    if not self._run_until_exhausted and self._cnt == self._num_steps:
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
    logging.debug('chainable: %s async dequeueing', self.name)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        self._thread_pool, _async_get_batch, self
    )
    logging.debug('chainable: "%s" async dequeued %d', self.name, len(result))
    return result

  async def async_get(self):
    """Gets an element from the queue."""
    logging.debug('chainable: %s async dequeueing', self.name)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(self._thread_pool, _async_get, self)

  async def async_put(self, value: _ValueT):
    logging.debug('chainable: %s async enqueueing a %s', self.name, type(value))
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
        logging.exception('chainable: %s enqueue failed.', self.name)
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
      try:
        if not self._cache:
          self._cache.extend(await self._iterator_queue.async_get_batch())
      except StopAsyncIteration as e:
        logging.info(
            'chainable: async iterator at %s exhausted after %d batches with'
            ' %d returns',
            getattr(self._iterator_queue, 'name', ''),
            self._cnt,
            len(e.args),
        )
        raise e
      self._cnt += 1
      name = getattr(self._iterator_queue, 'name', '')
      logging.debug(
          'chainable: %s', f'"{name}" async iter yield batch cnt: {self._cnt}'
      )
      return self._cache.popleft()
    else:
      raise StopAsyncIteration()

  def __aiter__(self):
    return self


def piter(
    iterator_fn: Callable[..., Iterable[_ValueT]],
    input_iterator: Iterable[Any] | None = None,
    *,
    max_parallism: int = 1,
    buffer_size: int = 0,
    thread_pool: futures.ThreadPoolExecutor | None = None,
) -> Iterable[_ValueT]:
  """Call a chain of functions in sequence concurrently with multithreads.

  Args:
    iterator_fn: A generator function that takes an iterable as input.
    input_iterator: The input iterator to be passed to the iterator_fn.
    max_parallism: The maximum number xof threads to be used.
    buffer_size: The buffer size of the queue.
    thread_pool: The thread pool to be used. If None, a new thread pool will be
      created.

  Returns:
    An iterable that iterates through the chain of functions.
  """
  output_queue = IteratorQueue(buffer_size, name='piter_output_q')
  input_q = None
  pool = thread_pool or futures.ThreadPoolExecutor()
  if input_iterator is not None:
    # The input_queue uses a max_batch_size of 1 to ensure that the output_queue
    # is not consuming too many elements that leads to imbalanced parallelism.
    max_batch_size = 1 if max_parallism > 1 else 0
    input_q = IteratorQueue(
        buffer_size, max_batch_size=max_batch_size, name='piter_input_q'
    )
    pool.submit(input_q.enqueue_from_iterator, input_iterator)
  for _ in range(max_parallism):
    if input_q is not None:
      it = iterator_fn(input_q)
    else:
      it = iterator_fn()
    pool.submit(output_queue.enqueue_from_iterator, it)
  return output_queue


def pmap(
    fn: Callable[..., Any],
    input_iterator: Iterable[Any] = (),
    *,
    max_parallism: int = 1,
    buffer_size: int = 0,
    thread_pool: futures.ThreadPoolExecutor | None = None,
):
  return piter(
      lambda x: map(fn, x),
      input_iterator,
      max_parallism=max_parallism,
      buffer_size=buffer_size,
      thread_pool=thread_pool,
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
    def wrapped_fn(*inputs, **kwargs) -> tuple[_ValueT, ...] | _ValueT:
      kwargs_keys = tuple(kwargs.keys())
      args_and_kwargs = itt.zip_longest(
          zip(*inputs), zip(*kwargs.values()), fillvalue=()
      )
      inputs = (
          (row_inputs, dict(zip(kwargs_keys, row_kwinputs)))
          for row_inputs, row_kwinputs in args_and_kwargs
      )
      if not thread_pool:
        outputs = list(
            fn(*row_inputs, **row_kwinputs)
            for row_inputs, row_kwinputs in inputs
        )
      else:
        states = []
        for row_inputs, row_kwinputs in inputs:
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
    num_columns: int,
    pad: Any = None,
) -> Iterator[tuple[_ValueT, ...]]:
  """Merges and concatenates n batches of tuples while iterating."""
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
