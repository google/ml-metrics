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
from ml_metrics._src.utils import iter_utils

pickler = lazy_fns.pickler


def setUpModule():
  testutil.SetupMockBNS()


class TestServer(courier_server.CourierServerWrapper):
  """Test server for CourierServerWrapper."""

  def set_up(self):
    super().set_up()

    def plus_one(x: int):
      return pickler.dumps(x + 1)

    assert self._server is not None
    self._server.Bind('plus_one', plus_one)


class CourierServerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server._cached_server()
    self.client = courier_worker.Worker(self.server.address)
    self.client.wait_until_alive(deadline_secs=12)

  def test_courier_server_maybe_make(self):
    client = courier.Client(self.server.address, call_timeout=1)
    self.assertEqual('hello', pickler.loads(client.maybe_make('hello')))
    result = client.maybe_make(pickler.dumps(lazy_fns.trace(len)([1, 2])))
    self.assertEqual(2, pickler.loads(result))

  def test_courier_server_custom_setup(self):
    server = TestServer()
    server.build_server()
    thread = server.start()
    client = courier.Client(server.address, call_timeout=2)
    self.assertEqual(2, pickler.loads(client.plus_one(1)))
    client.shutdown()
    thread.join()

  def test_courier_server_heartbeat(self):
    client = courier.Client(self.server.address, call_timeout=1)
    f = client.futures.heartbeat()
    self.assertIsNone(f.result())

  def test_courier_server_generator(self):
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

  def test_courier_server_batch_generator(self):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      yield from range(n)
      return n

    client.init_generator(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    while True:
      states = pickler.loads(client.next_batch_from_generator())
      if states and iter_utils.is_stop_iteration(states[-1]):
        actual.extend(states[:-1])
        returned = states[-1].value
        break
      else:
        actual.extend(states)
    self.assertEqual(list(range(10)), actual)
    self.assertEqual(10, returned)

  def test_courier_server_shutdown_and_restart(self):
    server = courier_server.CourierServerWrapper('test_restart')
    server.start()
    worker = courier_worker.Worker(server.address)
    worker.wait_until_alive(deadline_secs=12)
    assert server._thread is not None
    self.assertTrue(server._thread.is_alive())
    self.assertTrue(server.has_started)
    worker.shutdown()
    server._thread.join()
    # Restart.
    server.start()
    worker = courier_worker.Worker(server.address)
    worker.wait_until_alive(deadline_secs=12)
    self.assertEqual('echo', worker.get_result(lazy_fns.trace('echo')))
    worker.shutdown()
    server._thread.join()

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

  def test_make_remote_iterator(self):
    remote_iterator = courier_server.make_remote_iterator(
        range(10), server_addr=self.server.address
    )
    self.assertEqual(list(remote_iterator), list(range(10)))
    self.assertEqual(list(remote_iterator), [])

  def test_make_remote_iterator_lazy(self):
    remote_iterator = courier_server.make_remote_iterator(
        lazy_fns.trace(range)(10), server_addr=self.server.address
    )
    self.assertEqual(list(range(10)), list(remote_iterator))
    self.assertEqual([], list(remote_iterator))


if __name__ == '__main__':
  absltest.main()
