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
import courier
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import courier_utils
from ml_metrics._src.utils import iter_utils


class TimeoutServer(courier_server.PrefetchedCourierServer):
  """Test server for PrefetchedCourierServer."""

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


TIMEOUT_SERVER = TimeoutServer('timeout_server')
UNREACHABLE_SERVER = courier_server.CourierServer('unreachable_server')


def setUpModule():
  # Required for BNS resolution.
  testutil.SetupMockBNS()
  # Manually start and stop the server to avoid mocked courier BNS error.
  UNREACHABLE_SERVER.start()
  UNREACHABLE_SERVER.stop().join()
  TIMEOUT_SERVER.start(daemon=True)


def tearDownModule():
  TIMEOUT_SERVER.stop().join()


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


class RemoteObjectTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServer('RemoteObject')
    self.server.start()
    self.worker = courier_utils.CourierClient(self.server.address)

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
          get_result=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value,
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='remote_with_index',
          get_result=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value[0],
          expected=1,
      ),
      dict(
          testcase_name='remote_call',
          get_result=True,
          value=lazy_fns.trace(len)([1, 2, 3], lazy_result_=True),
          fn=lambda remote_value: remote_value,
          expected=3,
      ),
      dict(
          testcase_name='remote_attribute',
          get_result=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value.count(2),
          expected=1,
      ),
      dict(
          testcase_name='submit_remote_self',
          submit=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value,
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='submit_remote_with_index',
          submit=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value[0],
          expected=1,
      ),
      dict(
          testcase_name='submit_remote_call',
          submit=True,
          value=lazy_fns.trace(len)([1, 2, 3], lazy_result_=True),
          fn=lambda remote_value: remote_value,
          expected=3,
      ),
      dict(
          testcase_name='submit_remote_attribute',
          submit=True,
          value=lazy_fns.trace([1, 2, 3], lazy_result=True),
          fn=lambda remote_value: remote_value.count(2),
          expected=1,
      ),
  ])
  def test_maybe_make_remote_object(
      self, value, fn, expected, submit=False, get_result=False
  ):
    lazy_fns.clear_object()
    self.assertEqual(lazy_fns.object_info().currsize, 0)
    if submit:
      remote_value = courier_utils.RemoteObject.new(
          self.worker.submit(value).result(), worker=self.worker
      )
    elif get_result:
      remote_value = self.worker.get_result(value)
    else:
      remote_value = courier_utils.RemoteObject.new(value, worker=self.worker)
      if isinstance(value, lazy_fns.LazyObject):
        self.assertEqual(remote_value.id, value.id)

    self.assertIsInstance(remote_value, courier_utils.RemoteObject)
    # "RemoteObject" is the server name in `setUpModule`.
    self.assertRegex(str(remote_value), r'<@RemoteObject:object\(id=\d+\)>')
    self.assertEqual(lazy_fns.object_info().hits, 0)
    self.assertIsInstance(fn(remote_value), courier_utils.RemoteObject)
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
    self.assertIsInstance(remote_object, courier_utils.RemoteObject)
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
    remote_iterable = self.worker.get_result(
        lazy_fns.trace(range)(3, lazy_result_=True)
    )
    self.assertIsInstance(remote_iterable, courier_utils.RemoteObject)

    async def run():
      remote_iterator = aiter(remote_iterable)  # pytype: disable=name-error
      self.assertIsInstance(remote_iterator, courier_utils.RemoteIterator)
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
    input_iterator = courier_utils.RemoteIterator.new(
        local_queue.dequeue_as_iterator(), server_addr=self.server.address
    )
    iterator_fn = functools.partial(map, lambda x: x + 1)

    async def run():
      remote_iterator = await self.worker.async_iter(
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
    remote_iterable = self.worker.get_result(
        lazy_fns.trace(range)(3, lazy_result_=True)
    )
    self.assertIsInstance(remote_iterable, courier_utils.RemoteObject)
    remote_iterator = iter(remote_iterable)
    self.assertIsInstance(remote_iterator, courier_utils.RemoteIterator)
    self.assertIs(iter(remote_iterator), remote_iterator)
    self.assertEqual([0, 1, 2], list(remote_iterator))
    # 2nd iteration on an exhausted iterator.
    self.assertEqual([], list(remote_iterator))

    self.assertEqual([0, 1, 2], list(remote_iterable))
    # Iterable allows repeated traversing.
    self.assertEqual([0, 1, 2], list(remote_iterable))

  def test_remote_iterator_iterate_remotely(self):
    remote_iterable = self.worker.get_result(
        lazy_fns.trace(range)(3, lazy_result_=True)
    )
    actual = self.worker.get_result(lazy_fns.trace(list)(remote_iterable))
    self.assertEqual([0, 1, 2], actual)

  def test_remote_iterator_direct(self):
    # Reconstruct the iterator directly
    remote_iterable = courier_utils.RemoteObject.new(
        range(3), worker=self.worker
    )
    actual = self.worker.get_result(lazy_fns.trace(list)(remote_iterable))
    self.assertEqual([0, 1, 2], actual)

  def test_remote_iterator_queue_async(self):
    local_server = courier_server.CourierServer('local')
    local_server.start()
    remote_server = courier_server.CourierServer('remote')
    remote_server.start()
    # Constructs a local queue and let remote worker dequeue from it.
    local_queue = iter_utils.IteratorQueue(name='input')
    num_elem = 20

    def foo(n):
      yield from range(n)
      return n

    local_queue.enqueue_from_iterator(foo(num_elem))
    # input_iterator is remote and lives in local server.
    input_queue = courier_utils.RemoteIteratorQueue.new(
        local_queue, server_addr=local_server.address, name='remote_iter'
    )
    # Remotely constructs an iteraotor as the input_iterator.
    iterator_fn = functools.partial(map, lambda x: x + 1)
    lazy_iterator = lazy_fns.trace(iterator_fn)(lazy_fns.trace(input_queue))
    local_result_queue = iter_utils.AsyncIteratorQueue(
        timeout=30, name='output'
    )

    async def remote_iterate():
      remote_iterator = await courier_utils.async_remote_iter(
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
    local_server.stop().join()
    remote_server.stop().join()


class WorkerRegistryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServer('CourierWorker')
    self.server.start()
    self.worker = courier_utils.CourierClient(self.server.address)
    self.prefetched_server = courier_server.PrefetchedCourierServer()
    self.prefetched_server.start()
    self.prefetched_worker = courier_utils.CourierClient(
        self.prefetched_server.address
    )

  def test_client_registry(self):
    registry = courier_utils.WorkerRegistry()
    registry.register('a', 1.0)
    registry.refresh('b', 2.0)
    registry.refresh('a', 3.0)
    registry.register('c', 4.0)
    registry.unregister('b')
    registry.refresh('b', 5.0)
    self.assertEqual(registry.data, {'a': 3.0, 'c': 4.0, 'b': None})

  def test_client_registry_direct_set_raises(self):
    registry = courier_utils.WorkerRegistry()
    with self.assertRaisesRegex(TypeError, r'use register\(\) instead'):
      registry['a'] = 1.0

  def test_get_heartbeat(self):
    self.assertEqual(
        courier_utils.worker_registry().get('unknown_address'), 0.0
    )
    self.assertNotIn('fake_address', courier_utils.worker_registry())
    self.worker.send_heartbeat('fake_address').result()
    self.assertIn('fake_address', courier_utils.worker_registry())
    self.assertGreater(courier_utils.worker_registry().get('fake_address'), 0.0)
    self.worker.send_heartbeat('fake_address', is_alive=False).result()
    self.assertEqual(courier_utils.worker_registry().get('fake_address'), 0.0)

  @parameterized.named_parameters([
      dict(testcase_name='no_sender'),
      dict(testcase_name='alive', sender_addr='a'),
      dict(testcase_name='dead', sender_addr='a', is_alive=False),
  ])
  def test_heartbeat(self, sender_addr: str = '', is_alive: bool = True):
    client = courier.Client(self.server.address, call_timeout=1)
    f = client.futures.heartbeat(sender_addr, is_alive)
    self.assertIsNone(f.result())
    registry = courier_utils.worker_registry()
    if sender_addr and is_alive:
      self.assertIsNotNone(registry.get(sender_addr))
    else:
      self.assertEqual(registry.get(sender_addr), 0.0)


class CourierClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServer('CourierWorker')
    self.server.start()
    self.worker = courier_utils.CourierClient(self.server.address)
    self.prefetched_server = courier_server.PrefetchedCourierServer()
    self.prefetched_server.start()
    self.prefetched_worker = courier_utils.CourierClient(
        self.prefetched_server.address
    )

  def test_heartbeat_interval(self):
    if self.worker._heartbeat is None:
      self.worker._check_heartbeat()
    assert self.worker._heartbeat is not None
    first_heartbeat_time = self.worker._heartbeat.time
    time.sleep(0.2)
    self.worker._check_heartbeat(0.1)
    self.assertGreater(self.worker._heartbeat.time, first_heartbeat_time)
    self.assertEqual(
        self.worker._pendings[-1].time, self.worker._heartbeat.time
    )

  def test_worker_not_started(self):
    worker = courier_utils.CourierClient(
        UNREACHABLE_SERVER.address,
        heartbeat_threshold_secs=1,
        call_timeout=0.1,
    )
    with self.assertRaises(RuntimeError):
      worker.get_result(lazy_fns.trace(None))
    with self.assertRaises(RuntimeError):
      asyncio.run(worker.async_get_result(lazy_fns.trace(None)))

  def test_worker_get_result_timeout(self):
    worker = courier_utils.CourierClient(self.server.address, call_timeout=0.01)
    with self.assertRaises(TimeoutError):
      worker.get_result(lazy_fns.trace(time.sleep)(0.3))
    with self.assertRaises(TimeoutError):
      asyncio.run(worker.async_get_result(lazy_fns.trace(time.sleep)(0.3)))

  def test_worker_hash(self):
    addr = self.server.address
    d = {courier_utils.CourierClient(addr)}
    self.assertIn(courier_utils.CourierClient(addr), d)
    self.assertNotIn(courier_utils.CourierClient(addr, call_timeout=1), d)
    self.assertNotIn(courier_utils.CourierClient(addr, max_parallelism=2), d)
    self.assertNotIn(
        courier_utils.CourierClient(addr, heartbeat_threshold_secs=2), d
    )

  def test_worker_str(self):
    self.assertRegex(
        str(self.worker),
        r'CourierClient\("CourierWorker", timeout=.+,'
        r' from_last_heartbeat=.+\)',
    )

  def test_worker_submit_task(self):
    def foo():
      time.sleep(0.5)
      return 2

    task = courier_utils.Task.new(lazy_fns.trace(foo)(), blocking=True)
    task = self.worker.submit(task)
    self.assertTrue(task.done())
    self.assertEqual(2, task.result())

  def test_async_iterate(self):
    iterator_fn = functools.partial(map, lambda x: x + 1)
    lazy_iterator = lazy_fns.trace(iterator_fn)(range(3))
    remote_iterator = self.worker.async_iter(lazy_iterator)

    async def alist(remote_iterator):
      return [elem async for elem in await remote_iterator]

    self.assertEqual([1, 2, 3], asyncio.run(alist(remote_iterator)))

  def test_async_iterable(self):
    iterator_fn = functools.partial(map, lambda x: x + 1)
    remote_iterable = self.worker.get_result(
        lazy_fns.trace(iterator_fn)(range(3), lazy_result_=True)
    )

    async def run():
      remote_iterator = await courier_utils.async_remote_iter(
          remote_iterable, name='remote_iter'
      )
      return [elem async for elem in remote_iterator]

    self.assertEqual([1, 2, 3], asyncio.run(run()))

  def test_legacy_async_iterate(self):

    task = courier_utils.GeneratorTask.new(lazy_fns.trace(mock_generator)(3))
    agg_q = queue.SimpleQueue()
    batch_outputs = []

    async def run():
      async for elem in self.prefetched_worker.async_iterate(
          task, generator_result_queue=agg_q
      ):
        # During iteration, the worker is considered occupied.
        self.assertFalse(self.prefetched_worker.has_capacity)
        batch_outputs.append(elem)

    asyncio.run(run())
    self.assertEqual(list(range(3)), batch_outputs)
    self.assertEqual(3, agg_q.get())
    # After iteration, the worker should return the capacity.
    try:
      self.prefetched_worker.wait_until_alive(
          deadline_secs=1, check_capacity=True
      )
    except Exception:  # pylint: disable=broad-exception-caught
      self.fail('Worker is not idle after iteration.')

  def test_legacy_async_iterate_ignore_error(self):
    def bad_generator(n, exc_i):
      for elem in range(n):
        if elem == exc_i:
          raise ValueError('bad generator')
        yield elem

    # The generator on the server ignores error and stops at the 1st exception.
    task = courier_utils.GeneratorTask.new(lazy_fns.trace(bad_generator)(5, 3))
    agg_q = queue.SimpleQueue()

    async def run():
      result = []
      async for x in self.prefetched_worker.async_iterate(
          task, generator_result_queue=agg_q
      ):
        result.append(x)
      return result

    with self.assertRaisesRegex(ValueError, 'bad generator'):
      _ = asyncio.run(run())

  def test_worker_heartbeat(self):
    # Server is not started, thus it is never alive.
    worker = courier_utils.CourierClient(
        UNREACHABLE_SERVER.address,
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

  def test_worker_shutdown(self):
    server = courier_server.CourierServer()
    t = server.start()
    worker = courier_utils.CourierClient(server.address)
    worker.wait_until_alive(deadline_secs=12)
    self.assertTrue(worker.call(True))
    self.assertTrue(t.is_alive())
    worker.shutdown()
    ticker = time.time()
    while t.is_alive():
      time.sleep(0)
      if time.time() - ticker > 10:
        self.fail('Server is not shutdown after 10 seconds.')


if __name__ == '__main__':
  absltest.main()
