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
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import lazy_fns

pickler = lazy_fns.picklers.default


class CourierServerTest(absltest.TestCase):

  def test_courier_server_maybe_make(self):
    server_wrapper = courier_server.CourierServerWrapper()
    server = server_wrapper.build_server()
    server.Start()
    client = courier.Client(server.address, call_timeout=6)
    self.assertEqual('hello', pickler.loads(client.maybe_make('hello')))
    self.assertEqual(
        2, pickler.loads(client.maybe_make(lazy_fns.trace(len)([1, 2])))
    )
    server.Stop()

  def test_courier_server_generator(self):
    server_wrapper = courier_server.CourierServerWrapper()
    server = server_wrapper.build_server()
    server.Start()
    client = courier.Client(server.address, call_timeout=6)

    def test_generator(n):
      yield from range(n)

    client.maybe_make(pickler.dumps(lazy_fns.trace(test_generator)(10)))
    actual = []
    while not lazy_fns.is_stop_iteration(
        t := pickler.loads(client.next_from_generator())
    ):
      actual.append(t)
    self.assertEqual(list(range(10)), actual)
    server.Stop()

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

  def test_courier_exception_during_prefetch(self):
    server_wrapper = courier_server.CourierServerWrapper()
    server = server_wrapper.build_server()
    t = threading.Thread(target=server_wrapper.run_until_shutdown)
    t.start()
    client = courier.Client(server.address, call_timeout=6)

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
      client.shutdown()
      self.assertEqual(list(range(6)), actual)
    self.assertRegex(cm.output[0], '.*Traceback.*')


if __name__ == '__main__':
  absltest.main()
