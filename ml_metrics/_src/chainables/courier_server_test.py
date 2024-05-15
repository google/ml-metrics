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
"""Tests for courier_server."""

import threading
import time

from absl.testing import absltest
import courier
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns


pickler = lazy_fns.picklers.default


def setUpModule():
  testutil.SetupMockBNS()


class CourierServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server_wrapper = courier_server.CourierServerWrapper()
    self.server = self.server_wrapper.build_server()
    self.t = threading.Thread(target=self.server_wrapper.run_until_shutdown)
    self.t.start()
    self.client = courier_worker.Worker(self.server.address)
    self.client.wait_until_alive()

  def tearDown(self):
    self.client.shutdown()
    self.t.join()
    super().tearDown()

  def test_courier_server_maybe_make(self):
    client = courier.Client(self.server.address, call_timeout=1)
    self.assertEqual('hello', pickler.loads(client.maybe_make('hello')))
    self.assertEqual(
        2, pickler.loads(client.maybe_make(lazy_fns.trace(len)([1, 2])))
    )

  def test_courier_server_generator(self):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      yield from range(n)

    client.maybe_make(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    while not lazy_fns.is_stop_iteration(
        t := pickler.loads(client.next_from_generator())
    ):
      actual.append(t)
    self.assertEqual(list(range(10)), actual)

  def test_courier_server_batch_generator(self):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      yield from range(n)

    client.maybe_make(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    while True:
      states = pickler.loads(client.next_batch_from_generator())
      if states and lazy_fns.is_stop_iteration(states[-1]):
        actual.extend(states[:-1])
        break
      else:
        actual.extend(states)
    self.assertEqual(list(range(10)), actual)

  def test_courier_server_shutdown(self):
    server_wrapper = courier_server.CourierServerWrapper()
    server = server_wrapper.build_server()
    t = threading.Thread(target=server_wrapper.run_until_shutdown)
    t.start()
    client = courier.Client(server.address, call_timeout=6)
    self.assertTrue(client.maybe_make(True))
    self.assertTrue(t.is_alive())
    client.shutdown()
    time.sleep(7)
    self.assertFalse(t.is_alive())
    t.join()

  def test_courier_exception_during_prefetch(self):
    client = courier.Client(self.server.address, call_timeout=1)

    def test_generator(n):
      for i in range(n):
        if i > 5:
          raise ValueError('test exception.')
        yield i

    client.maybe_make(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    with self.assertLogs(level='ERROR') as cm:
      while not lazy_fns.is_stop_iteration(
          t := pickler.loads(client.next_from_generator())
      ):
        actual.append(t)
      self.assertEqual(list(range(6)), actual)
    self.assertRegex(cm.output[0], '.*Traceback.*')


if __name__ == '__main__':
  absltest.main()
