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
from absl.testing import absltest
from absl.testing import parameterized
import courier
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import courier_utils
from ml_metrics._src.utils import iter_utils

pickler = lazy_fns.pickler


def setUpModule():
  testutil.SetupMockBNS()


def tearDownModule():
  courier_server.shutdown_all()
  assert not courier_worker.Worker('test_server').pendings


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

  def test_get_heartbeat(self):
    self.worker.send_heartbeat('fake_address').result()
    self.assertGreater(courier_server.client_heartbeat('fake_address'), 0.0)

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

  def test_maybe_make(self):
    client = courier.Client(self.server.address, call_timeout=1)
    self.assertEqual('hello', pickler.loads(client.maybe_make('hello')))
    result = client.maybe_make(pickler.dumps(lazy_fns.trace(len)([1, 2])))
    self.assertEqual(2, pickler.loads(result))

  def test_custom_setup(self):
    server = TestServer()
    server.start()
    client = courier.Client(server.address, call_timeout=2)
    self.assertEqual(2, pickler.loads(client.plus_one(1)))

  @parameterized.named_parameters([
      dict(testcase_name='no_sender'),
      dict(testcase_name='alive', sender_addr='a'),
      dict(testcase_name='dead', sender_addr='a', is_alive=False),
  ])
  def test_heartbeat(self, sender_addr: str = '', is_alive: bool = True):
    client = courier.Client(self.server.address, call_timeout=1)
    f = client.futures.heartbeat(sender_addr, is_alive)
    self.assertIsNone(f.result())
    if sender_addr and is_alive:
      self.assertIsNotNone(self.server.client_heartbeat(sender_addr))
    else:
      self.assertEqual(self.server.client_heartbeat(sender_addr), 0.0)

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

  def test_generator(self):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      yield from range(n)

    client.init_generator(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    while not iter_utils.is_stop_iteration(
        t := pickler.loads(client.next_from_generator())
    ):
      actual.append(t)
    self.assertEqual(list(range(10)), actual)

  def test_batch_generator(self):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      yield from range(n)
      return n

    client.init_generator(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    while True:
      states = pickler.loads(client.next_batch_from_generator(2))
      if states and iter_utils.is_stop_iteration(states[-1]):
        actual.extend(states[:-1])
        returned = states[-1].value
        break
      else:
        actual.extend(states)
    self.assertEqual(list(range(10)), actual)
    self.assertEqual(10, returned)

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
          t := pickler.loads(client.next_from_generator()), Exception
      ):
        actual.append(t)
      self.assertEqual(list(range(6)), actual)
    self.assertRegex(cm.output[0], '.*Traceback.*')


if __name__ == '__main__':
  absltest.main()
