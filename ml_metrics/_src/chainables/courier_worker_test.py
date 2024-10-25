# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import functools
import queue
import time

from absl.testing import absltest
from absl.testing import parameterized
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import iter_utils
import portpicker

# For test, accelerate the heartbeat interval.
courier_worker._HRTBT_INTERVAL_SECS = 0.1
courier_worker._HRTBT_THRESHOLD_SECS = 1
Task = courier_worker.Task


def setUpModule():
  # Required for BNS resolution.
  testutil.SetupMockBNS()
  # Setup the server for the test group below.
  courier_server._cached_server('RemoteObject')
  courier_server._cached_server('CourierWorker')
  courier_server._cached_server('WorkerPool')


def tearDownModule():
  # Required for BNS resolution.
  testutil.SetupMockBNS()
  # Setup the server for the test group below.
  courier_server._cached_server('RemoteObject').stop().join()
  courier_server._cached_server('CourierWorker').stop().join()
  courier_server._cached_server('WorkerPool').stop().join()


def lazy_q_fn(n, stop=False):
  q = queue.SimpleQueue()
  for i in range(n):
    q.put(i)
  if stop:
    q.put(iter_utils.STOP_ITERATION)
  return q


def mock_generator(n, sleep_interval=0.0):
  for i in range(n):
    yield i
    time.sleep(sleep_interval)
  return n


class TimeoutServer(courier_server.CourierServerWrapper):
  """Test server for CourierServerWrapper."""

  def set_up(self):
    super().set_up()

    def timeout(x):
      time.sleep(60)
      result = lazy_fns.maybe_make(x)
      return lazy_fns.pickler.dumps(result)

    def timeout_init_iterator(x):
      del x
      time.sleep(60)

    assert self._server is not None
    self._server.Unbind('maybe_make')
    self._server.Bind('maybe_make', timeout)
    self._server.Unbind('init_iterator')
    self._server.Bind('init_iterator', timeout_init_iterator)


class RemoteObjectTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server._cached_server('RemoteObject')
    self.worker = courier_worker.cached_worker(self.server.address)
    self.server.wait_until_alive(deadline_secs=12)

  @parameterized.named_parameters([
      dict(
          testcase_name='self',
          value=[1, 2, 3],
          fn=lambda remote_value: remote_value,
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='with_index',
          value=[1, 2, 3],
          fn=lambda remote_value: remote_value[0],
          expected=1,
      ),
      dict(
          testcase_name='call',
          value=len,
          fn=lambda remote_value: remote_value([1, 2, 3]),
          expected=3,
      ),
      dict(
          testcase_name='attribute',
          value=[1, 2, 3],
          fn=lambda remote_value: remote_value.count(2),
          expected=1,
      ),
      dict(
          testcase_name='queue',
          value=lazy_q_fn(3),
          fn=lambda remote_value: remote_value.qsize(),
          expected=3,
      ),
      dict(
          testcase_name='traced_self',
          value=lazy_fns.trace([1, 2, 3]),
          fn=lambda remote_value: remote_value,
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='traced_with_index',
          value=lazy_fns.trace([1, 2, 3]),
          fn=lambda remote_value: remote_value[0],
          expected=1,
      ),
      dict(
          testcase_name='traced_call',
          value=lazy_fns.trace(len),
          fn=lambda remote_value: remote_value([1, 2, 3]),
          expected=3,
      ),
      dict(
          testcase_name='traced_attribute',
          value=lazy_fns.trace([1, 2, 3]),
          fn=lambda remote_value: remote_value.count(2),
          expected=1,
      ),
      dict(
          testcase_name='remote_self',
          submit=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value,
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='remote_with_index',
          submit=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value[0],
          expected=1,
      ),
      dict(
          testcase_name='remote_call',
          submit=True,
          value=lazy_fns.trace(len)([1, 2, 3], lazy_result_=True),
          fn=lambda remote_value: remote_value,
          expected=3,
      ),
      dict(
          testcase_name='remote_attribute',
          submit=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value.count(2),
          expected=1,
      ),
  ])
  def test_maybe_make_remote_object(self, value, fn, expected, submit=False):
    lazy_fns.clear_object()
    self.assertEqual(lazy_fns.object_info().currsize, 0)
    if submit:
      remote_value = self.worker.submit(value).result()
    else:
      remote_value = courier_worker.RemoteObject.new(value, worker=self.worker)
      if isinstance(value, lazy_fns.LazyObject):
        self.assertEqual(remote_value.id, value.id)

    self.assertIsInstance(remote_value, courier_worker.RemoteObject)
    # "RemoteObject" is the server name in `setUpModule`.
    self.assertRegex(str(remote_value), r'<@RemoteObject:object\(id=\d+\)>')
    self.assertEqual(lazy_fns.object_info().hits, 0)
    self.assertIsInstance(fn(remote_value), courier_worker.RemoteObject)
    self.assertEqual(expected, lazy_fns.maybe_make(fn(remote_value)))
    # These are scenario that the local object is cached and a remote value only
    # holds a reference.
    if not isinstance(value, lazy_fns.LazyObject) or value.lazy_result:
      self.assertEqual(lazy_fns.object_info().hits, 1)

  def test_get_result_raises(self):
    def foo():
      raise ValueError('foo')

    remote_object = self.worker.get_result(
        lazy_fns.trace(foo, lazy_result=True)
    )
    self.assertIsInstance(remote_object, courier_worker.RemoteObject)
    with self.assertRaisesRegex(ValueError, 'foo'):
      remote_object().result_()

  def test_async_get_result_stop_iteration(self):
    def foo():
      raise StopIteration('foo')

    async def run():
      remote_object = await self.worker.async_get_result(
          lazy_fns.trace(foo, lazy_result=True)
      )
      return await remote_object().async_result_()

    returned = None
    try:
      asyncio.run(run())
    except StopAsyncIteration as e:
      returned = e.args
    self.assertEqual(returned, ('foo',))

  def test_async_get_result_raises_value_error(self):
    def foo():
      raise ValueError('foo')

    with self.assertRaises(ValueError):
      asyncio.run(self.worker.async_get_result(lazy_fns.trace(foo)()))

  def test_async_remote_iterator_iterate_elemnwise(self):
    remote_iterable = self.worker.submit(
        lazy_fns.trace(range)(3, lazy_result_=True)
    ).result()
    self.assertIsInstance(remote_iterable, courier_worker.RemoteObject)

    async def run():
      remote_iterator = aiter(remote_iterable)  # pytype: disable=name-error
      self.assertIsInstance(remote_iterator, courier_worker.RemoteIterator)
      self.assertIs(aiter(remote_iterator), remote_iterator)  # pytype: disable=name-error
      return [elem async for elem in remote_iterator]

    self.assertEqual([0, 1, 2], asyncio.run(run()))
    # 2nd iteration on an exhausted iterator.
    self.assertEqual([0, 1, 2], asyncio.run(run()))

  def test_remote_iterator_as_input_iterator(self):
    # Constructs a local queue and let remote worker dequeue from it.
    local_queue = iter_utils.IteratorQueue(name='input')
    num_elem = 30
    local_queue.enqueue_from_iterator(range(num_elem))
    # input_iterator is remote and lives in local server.
    input_iterator = courier_server.make_remote_iterator(
        local_queue.dequeue_as_iterator(), server_addr=self.server.address
    )
    iterator_fn = functools.partial(map, lambda x: x + 1)

    async def run():
      remote_iterator = self.worker.async_iter(
          lazy_fns.trace(iterator_fn)(input_iterator), name='remote_iter'
      )
      return [elem async for elem in remote_iterator]

    with self.assertLogs(level='INFO') as cm:
      actual = asyncio.run(run())
    self.assertEqual(list(range(1, num_elem + 1)), actual)
    self.assertEqual([], list(local_queue))
    logs = [l for l in cm.output if f'exhausted after {num_elem} batches' in l]
    self.assertLen(logs, 1)

  def test_remote_iterator_iterate_elemnwise(self):
    remote_iterable = self.worker.submit(
        lazy_fns.trace(range)(3, lazy_result_=True)
    ).result()
    self.assertIsInstance(remote_iterable, courier_worker.RemoteObject)
    remote_iterator = iter(remote_iterable)
    self.assertIsInstance(remote_iterator, courier_worker.RemoteIterator)
    self.assertIs(iter(remote_iterator), remote_iterator)
    self.assertEqual([0, 1, 2], list(remote_iterator))
    # 2nd iteration on an exhausted iterator.
    self.assertEqual([], list(remote_iterator))

    self.assertEqual([0, 1, 2], list(remote_iterable))
    # Iterable allows repeated traversing.
    self.assertEqual([0, 1, 2], list(remote_iterable))

  def test_remote_iterator_iterate_remotely(self):
    remote_iterable = self.worker.submit(
        lazy_fns.trace(range)(3, lazy_result_=True)
    ).result()
    actual = self.worker.submit(lazy_fns.trace(list)(remote_iterable)).result()
    self.assertEqual([0, 1, 2], actual)

  def test_remote_iterator_direct(self):
    # Reconstruct the iterator directly
    remote_iterable = courier_worker.RemoteObject.new(
        range(3), worker=self.worker
    )
    actual = self.worker.submit(lazy_fns.trace(list)(remote_iterable)).result()
    self.assertEqual([0, 1, 2], actual)

  def test_remote_iterator_queue_async(self):
    local_server = courier_server._cached_server('local')
    remote_server = courier_server._cached_server('remote')
    # Constructs a local queue and let remote worker dequeue from it.
    local_queue = iter_utils.IteratorQueue(name='input')
    num_elem = 20

    def foo(n):
      yield from range(n)
      return n

    local_queue.enqueue_from_iterator(foo(num_elem))
    # input_iterator is remote and lives in local server.
    input_queue = courier_server.make_remote_queue(
        local_queue, server_addr=local_server.address, name='remote_iter'
    )
    # Remotely constructs an iteraotor as the input_iterator.
    iterator_fn = functools.partial(map, lambda x: x + 1)
    lazy_iterator = lazy_fns.trace(iterator_fn)(lazy_fns.trace(input_queue))
    local_result_queue = iter_utils.AsyncIteratorQueue(
        timeout=30, name='output'
    )

    async def remote_iterate():
      remote_iterator = await courier_worker.async_remote_iter(
          lazy_iterator, worker=remote_server.address, name='remote_iter'
      )
      await local_result_queue.async_enqueue_from_iterator(remote_iterator)

    async def alist(iterator):
      return [elem async for elem in iterator]

    async def run(n):
      aw_result = alist(local_result_queue)
      result, *_ = await asyncio.gather(
          aw_result, *(remote_iterate() for _ in range(n))
      )
      return result

    actual = asyncio.run(run(2))
    self.assertEqual([], list(local_queue))
    self.assertCountEqual(list(range(1, num_elem + 1)), actual)
    self.assertEqual(num_elem, local_result_queue.returned[0])


class CourierWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server._cached_server('CourierWorker')
    self.worker = courier_worker.cached_worker(self.server.address)

  def test_heartbeat_interval(self):
    first_heartbeat_time = self.worker._heartbeat.time
    while not courier_worker._is_heartbeat_stale(self.worker._heartbeat):
      time.sleep(0.1)
    self.worker._send_heartbeat()
    self.assertGreater(self.worker._heartbeat.time, first_heartbeat_time)
    self.assertEqual(
        self.worker._pendings[-1].time, self.worker._heartbeat.time
    )

  def test_worker_not_started(self):
    server = courier_server._cached_server('unknown_worker')
    server.wait_until_alive(deadline_secs=12)
    worker = courier_worker.cached_worker(
        'unknown_worker', heartbeat_threshold_secs=1, call_timeout=0.1
    )
    worker.shutdown()
    server._thread.join()
    with self.assertRaises(RuntimeError):
      worker.get_result(None)
    with self.assertRaises(RuntimeError):
      asyncio.run(worker.async_get_result(None))

  def test_worker_get_result_timeout(self):
    worker = courier_worker.cached_worker(
        self.server.address, call_timeout=0.01
    )
    with self.assertRaises(TimeoutError):
      worker.get_result(lazy_fns.trace(time.sleep)(0.3))
    with self.assertRaises(TimeoutError):
      asyncio.run(worker.async_get_result(lazy_fns.trace(time.sleep)(0.3)))

  def test_worker_str(self):
    self.assertRegex(
        str(self.worker),
        r'Worker\(CourierWorker, timeout=.+, from_last_heartbeat=.+\)',
    )

  def test_worker_call(self):
    self.assertEqual(
        ['echo'],
        courier_worker.get_results([self.worker.call('echo')]),
    )

  def test_task_done(self):
    task = Task.new('echo')
    self.assertFalse(task.done())
    task = self.worker.submit(task)
    courier_worker.wait([task])
    self.assertTrue(task.done())

  def test_worker_submit_task(self):
    def foo():
      time.sleep(0.5)
      return 2

    task = Task.new(lazy_fns.trace(foo)(), blocking=True)
    task = self.worker.submit(task)
    self.assertTrue(task.done())
    self.assertEqual(2, task.result())

  def test_wait_timeout(self):
    task = Task.new(lazy_fns.trace(time.sleep)(1))
    task = self.worker.submit(task)
    self.assertNotEmpty(courier_worker.wait([task], timeout=0).not_done)

  def test_async_iterate(self):
    iterator_fn = functools.partial(map, lambda x: x + 1)
    lazy_iterator = lazy_fns.trace(iterator_fn)(range(3))
    remote_iterator = self.worker.async_iter(lazy_iterator)

    async def alist(remote_iterator):
      return [elem async for elem in remote_iterator]

    self.assertEqual([1, 2, 3], asyncio.run(alist(remote_iterator)))

  def test_async_iterable(self):
    iterator_fn = functools.partial(map, lambda x: x + 1)
    remote_iterable = self.worker.get_result(
        lazy_fns.trace(iterator_fn)(range(3), lazy_result_=True)
    )

    async def run():
      remote_iterator = await courier_worker.async_remote_iter(
          remote_iterable, name='remote_iter'
      )
      return [elem async for elem in remote_iterator]

    self.assertEqual([1, 2, 3], asyncio.run(run()))

  def test_legacy_async_iterate(self):

    task = courier_worker.GeneratorTask.new(lazy_fns.trace(mock_generator)(3))
    agg_q = queue.SimpleQueue()
    batch_outputs = []

    async def run():
      async for elem in self.worker.async_iterate(
          task, generator_result_queue=agg_q
      ):
        # During iteration, the worker is considered occupied.
        self.assertFalse(self.worker.has_capacity)
        batch_outputs.append(elem)

    asyncio.run(run())
    self.assertEqual(list(range(3)), batch_outputs)
    self.assertEqual(3, agg_q.get())
    # After iteration, the worker should return the capacity.
    try:
      self.worker.wait_until_alive(deadline_secs=1, check_capacity=True)
    except Exception:  # pylint: disable=broad-exception-caught
      self.fail('Worker is not idle after iteration.')

  def test_legacy_async_iterate_raise(self):

    def bad_generator():
      for elem in range(3):
        if elem == 2:
          raise TypeError('bad generator')
        yield elem

    task = courier_worker.GeneratorTask.new(lazy_fns.trace(bad_generator)())
    agg_q = queue.SimpleQueue()

    async def run():
      async for _ in self.worker.async_iterate(
          task, generator_result_queue=agg_q
      ):
        pass

    with self.assertRaises(TypeError):
      asyncio.run(run())

  def test_worker_heartbeat(self):
    # Server is not started, thus it is never alive.
    worker = courier_worker.Worker(
        f'localhost:{portpicker.pick_unused_port()}',
        heartbeat_threshold_secs=0,
        call_timeout=0.01,
    )
    self.assertFalse(worker.is_alive)

  def test_worker_pendings(self):
    r = self.worker.call(lazy_fns.trace(time.sleep)(0.3))
    self.assertLen(self.worker.pendings, 1)
    # wait until the call is finished.
    assert r.result()
    self.assertEmpty(self.worker.pendings)

  def test_worker_idle(self):
    while not self.worker.has_capacity:
      time.sleep(0)
    self.assertTrue(self.worker.has_capacity)
    r = self.worker.call(lazy_fns.trace(time.sleep)(0.3))
    self.assertFalse(self.worker.has_capacity)
    # wait until the call is finished.
    assert r.result()
    self.assertTrue(self.worker.has_capacity)

  def test_worker_exception(self):
    state_futures = [self.worker.call(lazy_fns.trace(len)(0.3))]
    exceptions = courier_worker.get_exceptions(state_futures)
    self.assertLen(exceptions, 1)
    self.assertIsInstance(exceptions[0], Exception)

  def test_worker_timeout(self):
    self.worker.set_timeout(0.01)
    state = self.worker.call(lazy_fns.trace(time.sleep)(0.3))
    exceptions = courier_worker.get_exceptions([state])
    self.assertLen(exceptions, 1)
    self.assertIsInstance(exceptions[0], Exception)

  def test_worker_shutdown(self):
    server = courier_server.CourierServerWrapper()
    t = server.start()
    worker = courier_worker.cached_worker(server.address)
    worker.wait_until_alive(deadline_secs=12)
    self.assertTrue(worker.call(True))
    self.assertTrue(t.is_alive())
    worker.shutdown()
    ticker = time.time()
    while t.is_alive():
      time.sleep(0)
      if time.time() - ticker > 10:
        self.fail('Server is not shutdown after 10 seconds.')


class TestServer(courier_server.CourierServerWrapper):
  """Test server for CourierServerWrapper."""

  def set_up(self):
    super().set_up()

    def plus_one(x: int | bytes):
      if isinstance(x, bytes):
        x = lazy_fns.pickler.loads(x)
      return lazy_fns.pickler.dumps(x + 1)

    assert self._server is not None
    self._server.Bind('plus_one', plus_one)


class CourierWorkerPoolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server._cached_server('WorkerPool')
    self.always_timeout_server = TimeoutServer()
    self.timeout_server_thread = self.always_timeout_server.start(daemon=True)
    self.worker_pool = courier_worker.WorkerPool([self.server.address])
    self.unreachable_address = f'localhost:{portpicker.pick_unused_port()}'
    self.worker_pool.wait_until_alive(deadline_secs=15)

  @parameterized.named_parameters([
      dict(testcase_name='from_workers'),
      dict(
          testcase_name='with_different_timeout',
          workerpool_params=dict(call_timeout=1),
      ),
      dict(
          testcase_name='with_different_max_parallelism',
          workerpool_params=dict(max_parallelism=1),
      ),
      dict(
          testcase_name='with_different_heartbeat_threshold_secs',
          workerpool_params=dict(heartbeat_threshold_secs=1),
      ),
      dict(
          testcase_name='with_different_iterate_batch_size',
          workerpool_params=dict(iterate_batch_size=1),
      ),
  ])
  def test_worker_pool_construct_from_workers(self, workerpool_params=None):
    workerpool_params = workerpool_params or {}
    worker_pool = courier_worker.WorkerPool(
        self.worker_pool.all_workers, **workerpool_params
    )
    assert len(worker_pool.all_workers) == 1
    worker = worker_pool.all_workers[0]
    self.assertIsNot(worker, self.worker_pool.all_workers[0])
    for x in workerpool_params:
      self.assertEqual(getattr(worker, x), workerpool_params[x])

  def test_worker_pool_call(self):
    actual = self.worker_pool.call_and_wait('echo')
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertEqual(['echo'], actual)

  def test_worker_pool_call_with_method_in_task(self):
    server = TestServer()
    server.start(daemon=True)
    server.wait_until_alive(deadline_secs=15)
    worker_pool = courier_worker.WorkerPool([server.address])
    task = courier_worker.Task.new(1, courier_method='plus_one')
    # We only have one task, so just return the first element.
    self.assertEqual(2, worker_pool.run(task))
    worker_pool.shutdown()

  def test_worker_pool_iterate_lazy_generator(self):
    lazy_generators = (lazy_fns.trace(mock_generator)(3) for _ in range(30))
    courier_worker._LOGGING_INTERVAL_SEC = 0.01
    generator_result_queue = queue.SimpleQueue()
    # Adds a worker that is not reachable.
    worker_pool = courier_worker.WorkerPool(
        [
            self.always_timeout_server.address,
            self.unreachable_address,
            self.server.address,
        ],
        max_parallelism=1,
        call_timeout=0.5,
    )
    worker_pool.wait_until_alive(deadline_secs=12, minimum_num_workers=2)
    self.assertLen(worker_pool.workers, 2)
    with self.assertLogs(level='INFO') as cm:
      results = list(
          worker_pool.iterate(
              lazy_generators,
              generator_result_queue=generator_result_queue,
              num_total_failures_threshold=0,
          )
      )
    self.assertEmpty(worker_pool.acquired_workers)
    self.assertLen(results, 3 * 30)
    self.assertCountEqual(list(range(3)) * 30, results)
    self.assertEqual(31, generator_result_queue.qsize())
    actual_agg = []
    while not generator_result_queue.empty():
      actual_agg.append(generator_result_queue.get())
    self.assertEqual([3] * 30, actual_agg[:-1])
    self.assertNotEmpty([l for l in cm.output if 'progress' in l])

  def test_worker_pool_iterate_by_task(self):
    tasks = [lazy_fns.trace(mock_generator)(3)] * 5
    courier_worker._LOGGING_INTERVAL_SEC = 0.01
    generator_result_queue = queue.SimpleQueue()
    with self.assertLogs(level='INFO') as cm:
      results = [
          result
          for result in self.worker_pool.iterate(
              tasks,
              generator_result_queue=generator_result_queue,
              num_total_failures_threshold=0,
          )
      ]
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertNotEmpty([l for l in cm.output if 'progress' in l])
    self.assertLen(results, 3 * 5)
    self.assertCountEqual(list(range(3)) * 5, results)
    self.assertEqual(6, generator_result_queue.qsize())
    actual_agg = []
    while not generator_result_queue.empty():
      actual_agg.append(generator_result_queue.get())
    self.assertEqual([3] * 5, actual_agg[:-1])

  # TODO: b/349174267 - re-neable the test when this test does not hang when
  # exiting.
  # def test_worker_pool_iterate_invalid_iterator(self):
  #   invalid_iterators = [lazy_fns.trace(len)([3])]
  #   generator_result_queue = queue.SimpleQueue()
  #   iterator = self.worker_pool.iterate(
  #       invalid_iterators,
  #       num_total_failures_threshold=0,
  #       generator_result_queue=generator_result_queue,
  #   )
  #   with self.assertRaises(Exception):
  #     next(iterator)

  def test_worker_pool_run(self):
    tasks = lazy_fns.trace(len)([1, 2])
    actual = self.worker_pool.run(tasks)
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertEqual(2, actual)

  def test_worker_cache_info(self):
    self.worker_pool.run(lazy_fns.trace(len)((1, 2), cache_result_=True))
    hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.worker_pool.run(lazy_fns.trace(len)((1, 2), cache_result_=True))
    new_hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.assertEqual(new_hits - hits, 1)

  def test_worker_cache_by_id(self):
    # The list is not hashable, this will fall back to hashing by id.
    fn = lazy_fns.trace(len)([1, 2], cache_result_=True)
    self.worker_pool.run(fn)
    hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.worker_pool.run(fn)
    new_hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.assertEqual(new_hits - hits, 1)

  def test_worker_pool_idle_workers(self):
    worker_pool = courier_worker.WorkerPool([self.server.address])
    worker_pool.wait_until_alive(deadline_secs=15)
    idle_workers = worker_pool.idle_workers()
    self.assertLen(idle_workers, 1)
    idle_workers[0].call(lazy_fns.trace(time.sleep)(1))
    self.assertEmpty(worker_pool.idle_workers())

  def test_worker_pool_shutdown(self):
    server = courier_server.CourierServerWrapper()
    t = server.start()
    worker_pool = courier_worker.WorkerPool([server.address])
    worker_pool.wait_until_alive(deadline_secs=15)
    self.assertTrue(worker_pool.call_and_wait(True))
    worker_pool.shutdown()
    ticker = time.time()
    while t.is_alive():
      time.sleep(0)
      if time.time() - ticker > 10:
        self.fail('Server is not shutdown after 10 seconds.')

  def test_worker_pool_fail_to_start(self):
    server = courier_server._cached_server('bad_server')
    worker = courier_worker.Worker('bad_server', heartbeat_threshold_secs=1)
    worker.wait_until_alive(deadline_secs=12)
    if t := server.stop():
      t.join()

    worker_pool = courier_worker.WorkerPool(
        ['bad_server'],
        call_timeout=0.01,
        heartbeat_threshold_secs=1,
    )
    with self.assertRaisesRegex(ValueError, 'Failed to connect to minimum.*'):
      worker_pool.wait_until_alive(deadline_secs=1)

  def test_worker_pool_num_workers(self):
    addrs = [
        f'localhost:{portpicker.pick_unused_port()}',
        f'localhost:{portpicker.pick_unused_port()}',
    ]
    worker_pool = courier_worker.WorkerPool(addrs)
    self.assertEqual(2, worker_pool.num_workers)


if __name__ == '__main__':
  absltest.main()
