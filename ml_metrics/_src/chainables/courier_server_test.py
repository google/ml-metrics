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

import queue
import time

from absl.testing import absltest
from absl.testing import parameterized
import courier
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import iter_utils


pickler = lazy_fns.pickler


def lazy_q_fn(n, stop=True):
  q = queue.SimpleQueue()
  for i in range(n):
    q.put(i)
  if stop:
    q.put(iter_utils.STOP_ITERATION)
  return q


def setUpModule():
  testutil.SetupMockBNS()


class TestServer(courier_server.CourierServerWrapper):
  """Test server for CourierServerWrapper."""

  def set_up(self):
    super().set_up()

    def plus_one(x: int):
      return pickler.dumps(x + 1)

    self._server.Bind('plus_one', plus_one)


class CourierServerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServerWrapper()
    self.server.build_server()
    self.thread = self.server.start()
    self.client = courier_worker.Worker(self.server.address)
    self.client.wait_until_alive(deadline_secs=12)

  def tearDown(self):
    self.client.shutdown()
    self.thread.join()
    super().tearDown()

  def test_remote_object_self(self):
    remote_value = self.client.submit(
        lazy_fns.trace(len, remote=True)([1, 2, 3])
    ).result()
    self.assertIsInstance(remote_value, courier_server.RemoteObject)
    self.assertEqual(3, remote_value.value_())
    remote_value = remote_value.set_(gc=True)
    self.assertEqual(3, remote_value.value_())
    self.assertIsNone(remote_value.value_())

  def test_remote_object_with_index(self):
    remote_value = self.client.submit(
        lazy_fns.trace(tuple, remote=True)([1, 2, 3])
    ).result()
    self.assertIsInstance(remote_value, courier_server.RemoteObject)
    self.assertIsInstance(remote_value[1], courier_server.RemoteObject)
    self.assertEqual(2, remote_value[1].value_())

  def test_remote_object_with_attr(self):
    remote_value = self.client.submit(
        lazy_fns.trace(list, remote=True)([1, 2, 3])
    ).result()
    self.assertIsInstance(remote_value, courier_server.RemoteObject)
    self.assertIsInstance(remote_value.pop(), courier_server.RemoteObject)
    self.assertEqual(3, remote_value.pop().value_())

  def test_remote_object_queue(self):
    remote_queue = self.client.submit(
        lazy_fns.trace(lazy_q_fn, remote=True)(3)
    ).result()
    actual = []
    while not iter_utils.is_stop_iteration(
        value := remote_queue.get_nowait().value_()
    ):
      actual.append(value)
    self.assertEqual(actual, [0, 1, 2])

  def test_remote_queue_dequeue_normal(self):
    fns = [lazy_fns.trace(lazy_q_fn, remote=True)(2) for _ in range(3)]
    remote_qs = courier_server.RemoteQueues(
        self.client.submit(fn).result() for fn in fns
    )
    actual = list(remote_qs.dequeue())
    self.assertCountEqual(actual, [0, 0, 0, 1, 1, 1])

  def test_remote_queue_dequeue_timeout(self):
    fns = [
        lazy_fns.trace(lazy_q_fn, remote=True)(2, stop=False) for _ in range(3)
    ]
    remote_qs = courier_server.RemoteQueues(
        (self.client.submit(fn).result() for fn in fns), timeout_secs=1
    )
    with self.assertRaisesRegex(TimeoutError, 'Dequeue timeout'):
      list(remote_qs.dequeue())

  @parameterized.named_parameters([
      dict(
          testcase_name='normal',
          queue_total_size=12,
          input_size=3,
      ),
      dict(
          testcase_name='timeout_at_stop',
          queue_total_size=5,  # needs 3 (num_queues) stop + input_size = 6
          input_size=3,
      ),
      dict(
          testcase_name='timeout',
          queue_total_size=2,  # needs 3 (num_queues) stop + input_size = 6
          input_size=3,
      ),
  ])
  def test_remote_queue_enqueue(
      self,
      queue_total_size,
      input_size,
      num_queues=3,
  ):
    base_qsize, residual = divmod(queue_total_size, num_queues)
    q_sizes = [
        base_qsize + (1 if i < residual else 0) for i in range(num_queues)
    ]
    fns = [
        lazy_fns.trace(queue.Queue, remote=True)(maxsize=q_size)
        for q_size in q_sizes
    ]
    remote_qs = courier_server.RemoteQueues(
        (self.client.submit(fn).result() for fn in fns), timeout_secs=1
    )
    if queue_total_size >= input_size + num_queues:
      actual = list(remote_qs.enqueue(range(input_size)))
      self.assertCountEqual(actual, list(range(input_size)))
    else:
      with self.assertRaisesRegex(TimeoutError, 'Enqueue timeout'):
        list(remote_qs.enqueue(range(input_size)))

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
    time.sleep(2)
    self.assertTrue(f.done())

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

  def test_courier_server_shutdown(self):
    server = courier_server.CourierServerWrapper()
    server.build_server()
    t = server.start()
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
