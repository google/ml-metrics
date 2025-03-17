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

import asyncio
import collections
from collections.abc import Sequence
from concurrent import futures
import itertools as itt
import queue
import threading
import time

from absl.testing import absltest
from absl.testing import parameterized
from courier.python import testutil
from ml_metrics._src.utils import iter_utils
from ml_metrics._src.utils import test_utils
import more_itertools as mit
import numpy as np


async def alist(iterator):
  return [elem async for elem in iterator]


def setUpModule():
  # Required for BNS resolution.
  testutil.SetupMockBNS()


def args_batched(num_args, batch_size, batch_fn=lambda x: x):
  """Generates (tuple of) n batches of of the same value."""
  for _ in range(1000):
    yield tuple(batch_fn(np.ones(batch_size) * j) for j in range(num_args))
  raise ValueError(
      'Reached the end of the range, might indicate iterator is first exhausted'
      ' before running.'
  )


def iter_inc(it):
  for x in it:
    yield x + 1


class MockSequence:

  def __init__(self, data):
    self._data = data
    self.ref_cnt = 0

  def __len__(self):
    return len(self._data)

  def __getitem__(self, i):
    self.ref_cnt += 1
    return self._data[i.__index__()]


class IterUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.thread_pool = futures.ThreadPoolExecutor()

  def tearDown(self):
    self.thread_pool.shutdown()
    super().tearDown()

  def test_iterate_fn_with_broadcast(self):
    def foo(x, a=0):
      return x + 1 + a

    actual = iter_utils.iterate_fn(foo)([0, 1], a=2)
    self.assertEqual([3, 4], actual)
    actual = iter_utils.iterate_fn(foo)([0, 1], 2)
    self.assertEqual([3, 4], actual)
    actual = iter_utils.iterate_fn(foo)([0, 1])
    self.assertEqual([1, 2], actual)

  def test_parallel_iterate_fn_return_tuple(self):
    sleep_time = 0.6

    def foo(x, y):
      time.sleep(sleep_time)
      return x + 1, y + 2

    foo1 = iter_utils.iterate_fn(foo, multithread=16)
    foo = iter_utils.iterate_fn(multithread=16)(foo)
    self.assertIsNotNone(foo.thread_pool)
    # Same function with same maximum parallelism should share the same thread
    # pool.
    self.assertIs(foo.thread_pool, foo1.thread_pool)
    len1 = iter_utils.iterate_fn(multithread=16)(len)
    self.assertIs(len1.thread_pool, foo.thread_pool)
    foo2 = iter_utils.iterate_fn(multithread=8)(foo)
    self.assertIs(foo2.thread_pool, foo.thread_pool)

    ticker = time.time()
    np.testing.assert_array_equal(
        (np.ones(16), np.ones(16) * 3), foo(np.zeros(16), y=np.ones(16))
    )
    self.assertLess(time.time() - ticker, sleep_time * 2)

  def test_iterate_fn_return_tuple(self):
    def foo(x, y):
      return x + 1, y + 2

    self.assertEqual(
        ((1, 2), (4, 5)), iter_utils.iterate_fn(foo)([0, 1], y=[2, 3])
    )

  def test_iterate_fn_return_named_tuple(self):
    # Named tuple will not be transposed
    R = collections.namedtuple('R', ['a', 'b'])

    def foo(x):
      return R(x, 0)

    self.assertEqual([R(0, 0), R(1, 0)], iter_utils.iterate_fn(foo)([0, 1]))

  def test_iterate_ignore_error(self):
    it = iter_utils.index_slice(test_utils.SequenceWithExc(5, 3))
    it = map(lambda x: x, it)
    # 0, 1, 2, 3, 4, ignore 3, got 0, 1, 2, 4
    actual = list(iter_utils.iter_ignore_error(it))
    self.assertEqual([0, 1, 2, 4], actual)

  def test_map_ignore_error(self):
    def foo(x):
      if x == 2:
        raise ValueError('foo')
      return x

    it = iter_utils.map_ignore_error(foo, range(5))
    actual = list(it)
    self.assertEqual([0, 1, 3, 4], actual)

  def test_sequence_array_normal(self):
    a = iter_utils.SequenceArray(np.arange(10))
    self.assertIsInstance(a, Sequence)
    self.assertSequenceEqual(list(range(10)), a)
    self.assertLen(a, 10)
    self.assertEqual(a[9], 9)
    a[9] = 0
    self.assertEqual(a[9], 0)
    self.assertEqual(a.count(0), 2)
    self.assertEqual(a.index(1), 1)

  def test_sequence_array_index_raises(self):
    a = iter_utils.SequenceArray(np.arange(10))
    with self.assertRaisesRegex(ValueError, 'not in array'):
      a.index(10)

  def test_index_slice(self):
    seq = MockSequence(range(10))
    a = iter_utils.index_slice(seq, 3, 5)
    self.assertEqual(list(a), [3, 4])
    self.assertEqual(2, seq.ref_cnt)

  @parameterized.named_parameters([
      dict(
          testcase_name='default',
          seqs=[range(10), range(10, 20)],
      ),
      dict(
          testcase_name='non_sliceable_sequence',
          seqs=[MockSequence(range(10)), MockSequence(range(10, 20))],
      ),
  ])
  def test_merged_sequences_default(self, seqs):
    a = iter_utils.MergedSequences(seqs)
    self.assertLen(a, 20)
    self.assertEqual(list(a), list(range(20)))
    self.assertEqual(11, a[11])
    self.assertEqual(19, a[-1])
    self.assertEqual(list(a[0:11]), list(range(11)))
    self.assertEqual(list(a[:11]), list(range(11)))
    self.assertEqual(list(a[8:20]), list(range(8, 20)))
    self.assertEqual(list(a[11:]), list(range(11, 20)))

  def test_merged_sequences_sequences(self):
    seqs = [range(10), range(10, 20)]
    a = iter_utils.MergedSequences(seqs)
    self.assertEqual(a.sequences, seqs)

  def test_merged_sequences_index_raises(self):
    a = iter_utils.MergedSequences([range(10), range(10, 20)])
    with self.assertRaisesRegex(IndexError, 'Index -21 is out of range'):
      _ = a[-21]
    with self.assertRaisesRegex(IndexError, 'Index 20 is out of range'):
      _ = a[20]

  def test_multiplex_iterator_single_iterator(self):
    it = iter_utils.MultiplexIterator(data_sources=[range(10)])
    self.assertEqual(list(it), list(range(10)))
    self.assertEqual(type(it._iterator), type(iter(range(10))))

  def test_multiplex_iterator_single_iterator_with_iter_fn(self):
    it = iter_utils.MultiplexIterator(
        data_sources=[range(10)], iter_fn=iter_inc
    )
    self.assertEqual(list(it), list(range(1, 11)))

  def test_multiplex_iterator_parallel(self):
    it = iter_utils.MultiplexIterator(
        data_sources=[test_utils.range_with_sleep(10, 0.6), range(10, 20)],
        iter_fn=iter_inc,
        parallism=2,
    )
    self.assertCountEqual(list(it), list(range(1, 21)))
    self.assertLen(it._thread_pool._threads, 2)
    self.assertIsInstance(it._iterator, iter_utils.DequeueIterator)

  def test_multiplex_iterator_in_process(self):
    it = iter_utils.MultiplexIterator(
        data_sources=[range(10), range(10, 20)], parallism=0
    )
    self.assertCountEqual(list(it), list(range(20)))
    self.assertIsInstance(it._iterator, itt.chain)
    self.assertIsNone(it._thread_pool)

  def test_multiplex_iterator_raises(self):
    it = iter_utils.MultiplexIterator(
        data_sources=[test_utils.range_with_exc(10, 3)], parallism=1
    )
    with self.assertRaisesRegex(ValueError, 'range_with_exc'):
      _ = list(it)
    self.assertIsNotNone(it._thread_pool)
    self.assertNotEmpty(it._thread_pool._threads)
    self.assertTrue(all(not t.is_alive() for t in it._thread_pool._threads))

  def test_enqueue_dequeue_raises(self):
    q = iter_utils.IteratorQueue()
    q._exception = ValueError('foo')
    with self.assertRaises(ValueError):
      _ = list(q)

  def test_iterator_queue_enqueue_dequeue_single_thread(self):
    q = iter_utils.IteratorQueue()
    self.assertFalse(q.enqueue_done)
    q.enqueue_from_iterator(test_utils.range_with_return(10))
    self.assertTrue(q.enqueue_done)
    q_iter = q.dequeue_as_iterator(num_steps=10)
    actual = list(iter(q_iter))
    self.assertSequenceEqual(list(range(10)), actual)
    self.assertLen(q.returned, 1)
    self.assertEqual(10, q.returned[0])

  def test_iterator_queue_async_enqueue_dequeue_single_thread(self):
    q = iter_utils.AsyncIteratorQueue()
    # This merges two streams of iterator into one.
    q.enqueue_from_iterator(test_utils.range_with_return(1))
    q.enqueue_from_iterator(test_utils.range_with_return(2))
    output_q = iter_utils.AsyncIteratorQueue()
    asyncio.run(output_q.async_enqueue_from_iterator(q))
    async_iter = output_q.async_dequeue_as_iterator(num_steps=3)
    actual = asyncio.run(alist(aiter(async_iter)))
    self.assertSequenceEqual([0, 0, 1], actual)
    self.assertLen(output_q.returned, 2)
    self.assertEqual([1, 2], output_q.returned)

  def test_iterator_queue_multithread_enqueue_dequeue(self):
    n = 1024
    num_threads, num_dequeue_threads = 16, 16
    input_q = iter_utils.IteratorQueue(24, name='input')
    q = iter_utils.IteratorQueue(n, timeout=15, name='iter')
    for _ in range(num_threads):
      t = threading.Thread(target=q.enqueue_from_iterator, args=(input_q,))
      t.start()
    with futures.ThreadPoolExecutor() as thread_pool:
      states = [thread_pool.submit(list, q) for _ in range(num_dequeue_threads)]
      input_q.enqueue_from_iterator(test_utils.range_with_return(n))
    results = futures.as_completed(states)
    actual = list(itt.chain(*(result.result() for result in results)))
    expected = list(range(n))
    self.assertCountEqual(expected, actual)
    self.assertEqual([n] * num_threads, q.returned)

  def test_iterator_queue_mt_enqueue_async_dequeue(self):
    n = 1024
    input_q = iter_utils.IteratorQueue()
    num_threads, num_dequeue_threads = 16, 16
    q = iter_utils.AsyncIteratorQueue(24, timeout=15)
    for _ in range(num_threads):
      threading.Thread(target=q.enqueue_from_iterator, args=(input_q,)).start()

    async def dequeue():
      tasks = (alist(q) for _ in range(num_dequeue_threads))
      results = await asyncio.gather(*tasks)
      return list(itt.chain(*results))

    input_q.enqueue_from_iterator(test_utils.range_with_return(n))
    actual = asyncio.run(dequeue())
    expected = list(range(n))
    self.assertCountEqual(expected, actual)

  def test_enqueue_from_generator_timeout(self):
    q = iter_utils.IteratorQueue.from_queue(queue.Queue(1), timeout=0.1)
    with self.assertRaisesRegex(TimeoutError, 'Enqueue timeout='):
      q.enqueue_from_iterator(range(2))

  def test_dequeue_from_generator_timeout(self):
    q = iter_utils.IteratorQueue(queue_or_size=1, timeout=0.1)
    q._queue.put(1)
    with self.assertRaisesRegex(TimeoutError, 'dequeue timeout'):
      mit.last(q)

  def test_async_dequeue_from_generator_timeout(self):
    q = iter_utils.AsyncIteratorQueue(queue_or_size=1, timeout=0.1)
    with self.assertRaises(TimeoutError):
      asyncio.run(q.async_get())
    with self.assertRaises(TimeoutError):
      asyncio.run(alist(q))

  def test_async_enqueue_from_generator_raises(self):
    q = iter_utils.AsyncIteratorQueue(timeout=0.1)

    async def async_iter():
      await asyncio.sleep(10)
      yield 1

    with self.assertRaises(TimeoutError):
      asyncio.run(q.async_enqueue_from_iterator(async_iter()))

  def test_async_enqueue_full_raises(self):
    q = iter_utils.AsyncIteratorQueue(1, timeout=1)

    async def async_iter():
      for i in range(2):
        yield i

    with self.assertRaises(TimeoutError):
      asyncio.run(q.async_enqueue_from_iterator(async_iter()))

  def test_iterator_queue_flush_raises(self):
    iterator = iter_utils.IteratorQueue(2)
    with futures.ThreadPoolExecutor() as thread_pool:
      thread_pool.submit(
          iterator.enqueue_from_iterator, test_utils.range_with_exc(10, 9)
      )
      with self.assertRaises(ValueError):
        _ = iterator.get_batch(block=True)

  def test_iterator_queue_flush_normal(self):
    iterator = iter_utils.IteratorQueue(2)
    result = []
    with futures.ThreadPoolExecutor() as thread_pool:
      thread_pool.submit(
          iterator.enqueue_from_iterator, test_utils.range_with_return(10)
      )
      qsize = iterator._queue.qsize()
      result += iterator.get_batch()
      result += iterator.get_batch(2, block=True)
      result += iterator.get_batch(block=True)
    self.assertLessEqual(qsize, 2)
    self.assertEqual(list(range(10)), result)
    self.assertEqual([10], iterator.returned)

  def test_iterator_queue_flush_early_stop(self):
    iterator = iter_utils.IteratorQueue()
    num_examples = 3
    with futures.ThreadPoolExecutor() as thread_pool:
      thread_pool.submit(
          iterator.enqueue_from_iterator,
          test_utils.range_with_sleep(10, sleep=0.25),
      )
      while iterator._queue.qsize() < num_examples:
        time.sleep(0)
      iterator.stop_enqueue()
      result = iterator.get_batch(block=True)
    self.assertGreaterEqual(len(result), num_examples)
    self.assertLess(len(result), 10)

  @parameterized.named_parameters([
      dict(
          testcase_name='no_op',
          expected=[(np.zeros(2), np.ones(2))] * 5,
      ),
      dict(
          testcase_name='to_larger_batch',
          batch_size=3,
          expected=[(np.zeros(3), np.ones(3))] * 3 + [([0], [1])],
      ),
      dict(
          testcase_name='to_larger_batch_padded',
          batch_size=3,
          pad=-1,
          expected=[
              (np.zeros(3), np.ones(3)),
              (np.zeros(3), np.ones(3)),
              (np.zeros(3), np.ones(3)),
              (np.array([0, -1, -1]), np.array([1, -1, -1])),
          ],
      ),
      dict(
          testcase_name='batch_size_is_same',
          batch_size=2,
          expected=[
              (np.zeros(2), np.ones(2)),
          ]
          * 5,
      ),
      dict(
          testcase_name='to_smaller_batch',
          batch_size=1,
          expected=[
              (np.zeros(1), np.ones(1)),
          ]
          * 10,
      ),
      dict(
          testcase_name='to_smaller_batch_padded',
          input_batch_size=3,
          batch_size=2,
          pad=-1,
          expected=[(np.zeros(2), np.ones(2))] * 7 + [([0, -1], [1, -1])],
      ),
      dict(
          testcase_name='with_list',
          batch_size=4,
          batch_fn=list,
          expected=[
              ([0, 0, 0, 0], [1, 1, 1, 1]),
              ([0, 0, 0, 0], [1, 1, 1, 1]),
              ([0, 0], [1, 1]),
          ],
      ),
      dict(
          testcase_name='with_list_padded',
          batch_size=4,
          batch_fn=list,
          pad=-1,
          expected=[
              ([0, 0, 0, 0], [1, 1, 1, 1]),
              ([0, 0, 0, 0], [1, 1, 1, 1]),
              ([0, 0, -1, -1], [1, 1, -1, -1]),
          ],
      ),
      dict(
          testcase_name='with_tuple',
          batch_size=4,
          batch_fn=tuple,
          expected=[
              ((0, 0, 0, 0), (1, 1, 1, 1)),
              ((0, 0, 0, 0), (1, 1, 1, 1)),
              ((0, 0), (1, 1)),
          ],
      ),
      dict(
          testcase_name='with_tuple_padded',
          batch_size=4,
          batch_fn=tuple,
          pad=-1,
          expected=[
              ((0, 0, 0, 0), (1, 1, 1, 1)),
              ((0, 0, 0, 0), (1, 1, 1, 1)),
              ((0, 0, -1, -1), (1, 1, -1, -1)),
          ],
      ),
  ])
  def test_rebatched(
      self,
      expected,
      batch_size=0,
      batch_fn=lambda x: x,
      input_batch_size=2,
      pad=None,
  ):
    infinite_batches = args_batched(
        num_args=2, batch_size=input_batch_size, batch_fn=batch_fn
    )
    inputs = itt.islice(infinite_batches, 5)
    actual = iter_utils.rebatched_args(
        inputs, batch_size=batch_size, num_columns=2, pad=pad
    )
    for a, b in zip(expected, actual, strict=True):
      np.testing.assert_array_almost_equal(a, b)

  def test_batch_non_sequence_type(self):
    inputs = [(1, 2), (3, 4)]
    with self.assertRaisesRegex(TypeError, 'Non sequence type'):
      next(iter_utils.rebatched_args(iter(inputs), batch_size=4, num_columns=2))

  def test_batch_unsupported_type(self):
    inputs = [('aaa', 'bbb'), ('aaa', 'bbb')]
    with self.assertRaisesRegex(TypeError, 'Unsupported container type'):
      next(iter_utils.rebatched_args(iter(inputs), batch_size=4, num_columns=2))

  def test_recitable_iterator_normal(self):
    inputs = range(3)
    it_inputs = iter_utils._TeeIterator(iter(inputs))
    it_outputs = map(lambda x: x + 1, it_inputs)
    actual = list(zip(it_outputs, it_inputs.tee(), strict=True))
    self.assertEqual([(1, 0), (2, 1), (3, 2)], actual)

  def test_recitable_iterator_raises(self):
    inputs = range(30)
    it_inputs = iter_utils._TeeIterator(iter(inputs), buffer_size=1)
    with self.assertRaisesRegex(IndexError, 'No element left'):
      list(it_inputs.tee())
    with self.assertRaisesRegex(RuntimeError, 'Buffer reached capacity'):
      _ = list(it_inputs)

  @parameterized.named_parameters([
      dict(
          testcase_name='to_larger_batch',
          input_batch_size=2,
          fn_batch_size=5,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='to_smaller_batch',
          input_batch_size=5,
          fn_batch_size=2,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='to_same_batch_size',
          input_batch_size=3,
          fn_batch_size=3,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='to_one_element_batch',
          input_batch_size=2,
          fn_batch_size=1,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='from_one_element_batch',
          input_batch_size=1,
          fn_batch_size=5,
          num_columns=2,
          num_batches=30,
      ),
  ])
  def test_recitable_iterator_with_rebatch(
      self,
      input_batch_size=2,
      fn_batch_size=3,
      num_columns=2,
      num_batches=5,
  ):

    inputs = itt.islice(
        args_batched(num_columns, batch_size=input_batch_size), num_batches
    )

    def foo(columns):
      assert len(columns[0]) <= fn_batch_size, f'got {columns=}.'
      return tuple(np.array(column) + 1 for column in columns)

    def process_generator(it_inputs):
      it_fn_inputs = iter_utils.rebatched_args(
          it_inputs, batch_size=fn_batch_size, num_columns=num_columns
      )
      yield from iter_utils.rebatched_args(
          map(foo, it_fn_inputs),
          batch_size=input_batch_size,
          num_columns=num_columns,
      )

    # Setting a max buffer size to make sure the buffer is flushed while
    # iterating.
    actual = iter_utils.processed_with_inputs(process_generator, inputs)
    outputs, original = zip(*actual)

    expected_orignal = list(
        itt.islice(
            args_batched(num_columns, batch_size=input_batch_size), num_batches
        )
    )
    expected_outputs = [np.array(x) + 1 for x in expected_orignal]
    np.testing.assert_array_equal(expected_orignal, original)
    np.testing.assert_array_equal(expected_outputs, outputs)

  def test_iterator_queue_exception_all_threads_stop(self):
    q = iter_utils.IteratorQueue()
    num_threads = 5
    with futures.ThreadPoolExecutor() as thread_pool:
      for _ in range(num_threads):
        thread_pool.submit(
            q.enqueue_from_iterator, test_utils.range_with_sleep(999, 0.2)
        )
      thread_pool.submit(
          q.enqueue_from_iterator, test_utils.range_with_exc(10, 3)
      )
      self.assertLen(thread_pool._threads, num_threads + 1)
      with self.assertRaisesRegex(ValueError, 'range_with_exc at 3'):
        _ = list(q)
    # This is more for documentation purpose since all threads should have
    # stopped when thread pool is destroyed.
    self.assertTrue(all(not t.is_alive()for t in thread_pool._threads))

  def test_iterator_queue_ignore_error_with_skippable_input(self):
    q = iter_utils.IteratorQueue(ignore_error=True)
    with futures.ThreadPoolExecutor() as thread_pool:
      f = thread_pool.submit(list, q)
      thread_pool.submit(
          q.enqueue_from_iterator,
          iter_utils.index_slice(test_utils.SequenceWithExc(4, 2)),
      )
      self.assertEqual([0, 1, 3], f.result())

  def test_iterator_queue_ignore_error_with_non_skippable_input(self):
    q = iter_utils.IteratorQueue(ignore_error=True)
    with futures.ThreadPoolExecutor() as thread_pool:
      f = thread_pool.submit(list, q)
      thread_pool.submit(
          q.enqueue_from_iterator, test_utils.range_with_exc(5, 2)
      )
      self.assertEqual([0, 1], f.result())

  def test_piter_fn_only(self):
    n = 256
    with futures.ThreadPoolExecutor() as thread_pool:
      pit = iter_utils.piter_fn(
          iter_inc,
          input_iterable=range(n),
          parallism=8,
          thread_pool=thread_pool,
      )
      actual = list(pit)
      self.assertCountEqual(list(range(1, n + 1)), actual)
    self.assertIsInstance(pit, iter_utils.IteratorQueue)

  def test_piter_piter_multiplex(self):
    n, m = 256, 2
    assert n % m == 0
    inputs = [test_utils.range_with_sleep(m, 0.6) for _ in range(int(n / m))]
    with futures.ThreadPoolExecutor(max_workers=len(inputs)) as thread_pool:
      pit = iter_utils.piter_multiplex(inputs, thread_pool=thread_pool)
      actual = list(pit)
      self.assertLen(thread_pool._threads, len(inputs))
    expected = list(itt.chain(*[range(m) for _ in range(int(n / m))]))
    self.assertCountEqual(expected, actual)
    self.assertIsInstance(pit, iter_utils.IteratorQueue)

  def test_piter_piter_multiplex_fast(self):
    n, m = 256, 2
    assert n % m == 0
    inputs = [range(m) for _ in range(int(n / m))]
    # This is to test when not all enqueuer are started before the dequeuer is,
    # which could cause premature StopIteration controled by `enqueue_done`.
    with futures.ThreadPoolExecutor(max_workers=1) as pool:
      pit = iter_utils.piter_multiplex(inputs, thread_pool=pool)
      actual = list(pit)
    expected = list(itt.chain(*inputs))
    self.assertCountEqual(expected, actual)
    self.assertIsInstance(pit, iter_utils.IteratorQueue)

  def test_piter_no_input_iterators_and_inputs_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'iterator_fn or input_iterators has to be provided.'
    ):
      _ = iter_utils.piter(None)

  def test_piter_input_iterators_and_iterate_fn(self):
    inputs = [range(3)] * 2
    inc = lambda x: x + 1
    it = iter_utils.piter(lambda x: map(inc, x), input_iterators=inputs)
    expected = list(map(inc, itt.chain(*inputs)))
    self.assertCountEqual(expected, list(it))

  def test_piter_multiple_and_concurrent_piter_fn(self):
    n, m = 256, 2
    assert n % m == 0
    p = n // m

    def foo(vals):
      for x in vals:
        time.sleep(0.3)
        yield x

    inputs = [test_utils.range_with_sleep(m, 0.3) for _ in range(p)]
    expected = list(itt.chain(*[range(m) for _ in range(p)]))
    with futures.ThreadPoolExecutor(max_workers=len(inputs) + p) as pool:
      it_input = iter_utils.piter_multiplex(
          inputs, thread_pool=pool, max_batch_size=1
      )
      it = iter_utils.piter_fn(
          foo, input_iterable=it_input, thread_pool=pool, parallism=p
      )
      self.assertIsInstance(it, iter_utils.IteratorQueue)
      actual = list(it)
      self.assertCountEqual(expected, actual)

  def test_pmap(self):

    def foo(x):
      # Sleep here 0.3s x 256 = 76.8s will make the small test timeout. This
      # test will be flaky as a result.
      time.sleep(0.3)
      return x + 1

    n = 256
    it = iter_utils.pmap(foo, range(n), max_parallism=256)
    actual = list(it)
    self.assertCountEqual(list(range(1, n + 1)), actual)
    self.assertIsInstance(it, iter_utils.IteratorQueue)

  def test_pmap_in_process(self):

    def foo(x):
      return x + 1

    n = 256
    it = iter_utils.pmap(foo, range(n), max_parallism=0)
    actual = list(it)
    self.assertCountEqual(list(range(1, n + 1)), actual)
    self.assertIsInstance(it, map)

  def test_pmap_early_stop(self):
    m, p = 2, 128  # p = 256 / m

    inputs = [test_utils.range_with_sleep(m, 0.3) for _ in range(p)]
    with futures.ThreadPoolExecutor(max_workers=len(inputs) + p) as pool:
      q = iter_utils.piter_multiplex(inputs, thread_pool=pool, max_batch_size=1)
      self.assertIsInstance(q, iter_utils.IteratorQueue)
      it = iter(q)
      _ = [next(it), next(it)]
      it.maybe_stop()
    # This assertion is for documentation purpose since the threads should have
    # stopped when the thread pool is destroyed.
    self.assertTrue(all(not t.is_alive() for t in pool._threads))


if __name__ == '__main__':
  absltest.main()
