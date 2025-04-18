# Copyright 2025 Google LLC
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

import threading
import time
from absl.testing import absltest
from absl.testing import parameterized
import courier
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import courier_utils
from ml_metrics._src.utils import iter_utils
from ml_metrics._src.utils import test_utils

pickler = lazy_fns.pickler


def setUpModule():
  testutil.SetupMockBNS()


def tearDownModule():
  courier_server.shutdown_all()


class TestServer(courier_server.CourierServer):
  """Test server for CourierServer."""

  def set_up(self):
    super().set_up()

    def plus_one(x: int):
      return pickler.dumps(x + 1)

    assert self._server is not None
    self._server.Bind('plus_one', plus_one)


class CourierServerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Both the client's host and the server are co-located, thus the address
    # is same for clients as well.
    self.server = courier_server.CourierServer(
        'test_server', clients=['test_server']
    )
    self.server.start()
    self.worker = courier_utils.CourierClient(self.server.address)

  @parameterized.named_parameters([
      dict(
          testcase_name='unnamed',
          server_name=None,
          expected_regex=r'CourierServer\(\"@localhost\:\d+\"\)',
      ),
      dict(
          testcase_name='named',
          server_name='named_server',
          expected_regex=r'CourierServer\(\"named_server@localhost\:\d+\"\)',
      ),
  ])
  def test_str(self, server_name: str | None, expected_regex: str | None):
    server = courier_server.CourierServer(server_name)
    self.assertRegex(str(server), expected_regex)

  @parameterized.named_parameters([
      dict(testcase_name='unnamed', config1={}, config2={}, equal=False),
      dict(
          testcase_name='same_names',
          config1=dict(server_name='hash_server1'),
          config2=dict(server_name='hash_server1'),
          equal=True,
      ),
      dict(
          testcase_name='same_names_different_timeouts',
          config1=dict(server_name='hash_server1', timeout_secs=10),
          config2=dict(server_name='hash_server1', timeout_secs=1),
      ),
      dict(
          testcase_name='same_names_different_prefetch_sizes',
          config1=dict(server_name='hash_server1'),
          config2=dict(server_name='hash_server1', prefetch_size=1),
      ),
      dict(
          testcase_name='different_names',
          config1=dict(server_name='hash_server2'),
          config2=dict(server_name='hash_server1'),
      ),
  ])
  def test_hash(self, config1, config2, equal=False):
    server = courier_server.PrefetchedCourierServer(**config1)
    server_set = {server}
    # This createes a new server with a different port.
    server1 = courier_server.PrefetchedCourierServer(**config2)
    if equal:
      self.assertIn(server1, server_set)
    else:
      self.assertNotIn(server1, server_set)

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='local_dead', local=True, addr='w1', alive=False),
      dict(testcase_name='local_notify', local=True, addr='w2'),
      dict(testcase_name='remote', local=False),
      dict(testcase_name='remote_notify', local=False, addr='w3'),
      dict(testcase_name='remote_dead', local=False, addr='w4', alive=False),
  ])
  def test_heartbeat(self, local: bool, addr: str = '', alive=True):
    client = courier.Client(self.server.address, call_timeout=1)
    if local:
      result = self.server._heartbeat(addr, is_alive=alive)
    else:
      result = client.heartbeat(addr, is_alive=alive)
    self.assertIsNone(result)
    if addr:
      self.assertIn(addr, courier_utils.worker_registry())
      if not alive:
        self.assertIsNone(courier_utils.worker_registry()[addr])

  def test_maybe_make(self):
    client = courier.Client(self.server.address, call_timeout=1)
    self.assertEqual('hello', pickler.loads(client.maybe_make('hello')))
    result = client.maybe_make(pickler.dumps(lazy_fns.trace(len)([1, 2])))
    self.assertEqual(2, pickler.loads(result))

  @parameterized.named_parameters([
      dict(testcase_name='pickled', lazy=False),
      dict(testcase_name='pickled_lazy', lazy=True),
  ])
  def test_maybe_make_local(self, lazy: bool):
    payload = 2
    if lazy:
      payload = lazy_fns.trace(len)([1, 2])
    result = pickler.loads(self.server._maybe_make(payload))
    self.assertEqual(2, result)

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='remote', local=False),
  ])
  def test_maybe_make_return_immediately(self, local: bool):
    client = courier.Client(self.server.address, call_timeout=1)
    start = time.time()
    payload = pickler.dumps(lazy_fns.trace(time.sleep)(0.5))
    if local:
      result = self.server._maybe_make(payload, return_immediately=True)
    else:
      result = client.maybe_make(payload, return_immediately=True)
    # Ignore result call should return immediately.
    self.assertLess(time.time() - start, 0.5)
    self.assertIsNone(pickler.loads(result))

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='remote', local=False),
  ])
  def test_maybe_make_return_none(self, local):
    class Foo:

      def __init__(self):
        self.x = 0

      def __call__(self):
        self.x += 1
        return self.x

    foo = lazy_fns.trace(Foo)(cache_result_=True)()
    client = courier.Client(self.server.address, call_timeout=1)
    if local:
      result = self.server._maybe_make(pickler.dumps(foo), return_none=True)
    else:
      result = client.maybe_make(pickler.dumps(foo), return_none=True)
    self.assertIsNone(pickler.loads(result))
    result = client.maybe_make(pickler.dumps(foo))
    self.assertEqual(2, pickler.loads(result))

  def test_custom_setup(self):
    server = TestServer()
    server.start()
    client = courier.Client(server.address, call_timeout=2)
    self.assertEqual(2, pickler.loads(client.plus_one(1)))

  def test_shutdown_and_restart(self):
    server = courier_server.CourierServer('test_restart')
    server.start()
    courier_worker.wait_until_alive(server.address, deadline_secs=12)
    assert server._thread is not None
    self.assertTrue(server._thread.is_alive())
    self.assertTrue(server.has_started)
    server.stop().join()
    # Restart.
    server.start()
    courier_worker.wait_until_alive(server.address, deadline_secs=12)

  def test_make_remote_iterator(self):
    remote_iterator = courier_utils.RemoteIterator.new(
        range(10), server_addr=self.server.address
    )
    self.assertEqual(list(remote_iterator), list(range(10)))
    self.assertEqual(list(remote_iterator), [])

  def test_make_remote_iterator_lazy(self):
    remote_iterator = courier_utils.RemoteIterator.new(
        lazy_fns.trace(range)(10), server_addr=self.server.address
    )
    self.assertEqual(list(range(10)), list(remote_iterator))
    self.assertEqual([], list(remote_iterator))


class PrefetchedCourierServerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.PrefetchedCourierServer(
        'generator_server', clients=['generator_server']
    )
    self.server.start()

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='remote', local=False),
  ])
  def test_generator(self, local: bool):
    client = courier.Client(self.server.address, call_timeout=1)

    iterator = pickler.dumps(lazy_fns.trace(range)(10))
    if local:
      self.server._init_iterator(iterator)
    else:
      client.init_generator(iterator)
    actual = []
    while not iter_utils.is_stop_iteration(
        t := pickler.loads(client.next_batch_from_generator(1))[0]
    ):
      actual.append(t)
    self.assertEqual(list(range(10)), actual)

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='remote', local=False),
  ])
  def test_batch_generator(self, local: bool):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      yield from range(n)
      return n

    client.init_generator(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    while True:
      if local:
        batch = pickler.loads(self.server._next_batch(2))
      else:
        batch = pickler.loads(client.next_batch_from_generator(2))
      if batch and iter_utils.is_stop_iteration(batch[-1]):
        actual.extend(batch[:-1])
        returned = batch[-1].value
        break
      else:
        actual.extend(batch)
    self.assertEqual(list(range(10)), actual)
    self.assertEqual(10, returned)

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='remote', local=False),
  ])
  def test_batch_generator_with_timeout(self, local: bool):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      yield from range(n)
      return n

    client.init_generator(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    # Simulate the case where the generator is killed.
    self.server._generator = None
    if local:
      states = pickler.loads(self.server._next_batch(2))
    else:
      states = pickler.loads(client.next_batch_from_generator(2))
    assert len(states) == 1
    self.assertIsInstance(states[-1], TimeoutError)

  def test_init_generator_with_shutdown(self):
    server = courier_server.PrefetchedCourierServer(prefetch_size=1)
    server.start()
    courier_worker.wait_until_alive(server.address, deadline_secs=12)
    client = courier.Client(server.address, call_timeout=1)
    server._shutdown_requested = True
    # When the worker is shutdown, any exception is converted into a Timeout.
    lazy_iter = lazy_fns.trace(test_utils.range_with_exc)(10, 8)
    state = client.init_generator(pickler.dumps(lazy_iter))
    self.assertIsInstance(state, TimeoutError)

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='remote', local=False),
  ])
  def test_batch_generator_with_shutdown(self, local: bool):
    # When the worker is shutdown, any exception is converted into a Timeout.
    server = courier_server.PrefetchedCourierServer(prefetch_size=1)
    server.start()
    courier_worker.wait_until_alive(server.address, deadline_secs=12)
    client = courier.Client(server.address, call_timeout=1)
    lazy_iter = lazy_fns.trace(test_utils.range_with_exc)(10, 8)
    server._init_iterator(pickler.dumps(lazy_iter))
    # Invalid generator will cause an exception when fetching the next batch.
    states = lazy_fns.pickler.loads(server._next_batch(6))
    self.assertEqual([0, 1, 2, 3, 4, 5], states)
    # Simulate the case where the worker is shutdown.
    if local:
      server._shutdown_requested = True
      states = pickler.loads(server._next_batch(6))
      self.assertLen(states, 1)
      self.assertIsInstance(states[-1], TimeoutError)
    else:
      future = client.futures.next_batch_from_generator(6)
      server._shutdown_requested = True
      # It is possible that the server still gets shutdown before the request
      # can be processed, in this case, we give it a free pass and rely on
      # the local test version to cover the logic.
      try:
        states = pickler.loads(future.result())
        self.assertLen(states, 1)
        self.assertIsInstance(states[-1], (ValueError, TimeoutError))
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  def test_courier_exception_during_prefetch(self):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      for i in range(n):
        if i > 5:
          raise ValueError('test exception.')
        yield i

    client.init_generator(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    with self.assertLogs(level='ERROR') as cm:
      while not isinstance(
          t := pickler.loads(client.next_batch_from_generator(1))[0], Exception
      ):
        actual.append(t)
      self.assertEqual(list(range(6)), actual)
    self.assertRegex(cm.output[0], '.*Traceback.*')

  def test_init_generator_lock(self):
    start, end = threading.Event(), threading.Event()

    def delayed_generator(n):
      start.set()
      time.sleep(1)
      end.set()
      return range(n)

    client = courier.Client(self.server.address)
    generator = lazy_fns.trace(delayed_generator)(10)
    t = threading.Thread(target=self.server._init_iterator, args=(generator,))
    t.start()
    generator = pickler.dumps(lazy_fns.trace(range)(10))
    # The second init_generator call have to wait until the first one finishes.
    start.wait()
    self.assertIsNone(client.init_generator(generator))
    self.assertTrue(end.is_set())

  @parameterized.named_parameters([
      dict(testcase_name='local', local=True),
      dict(testcase_name='remote', local=False),
  ])
  def test_stop_prefetch(self, local: bool):
    client = courier.Client(self.server.address, call_timeout=1)
    generator = pickler.dumps(lazy_fns.trace(test_utils.range_with_return)(10))
    assert client.init_generator(generator) is None
    self.assertIsNotNone(self.server._generator)
    if local:
      self.server._stop_prefetch()
    else:
      assert client.stop_prefetch() is None
    thread = self.server._enqueue_thread
    assert thread is not None
    self.assertFalse(thread.is_alive())
    assert self.server._generator is not None
    self.assertTrue(self.server._generator.exhausted)
    self.assertIsInstance(self.server._generator.exception, TimeoutError)


if __name__ == '__main__':
  absltest.main()
