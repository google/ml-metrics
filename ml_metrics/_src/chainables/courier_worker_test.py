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
import threading
import time

from absl.testing import absltest
from courier.python import testutil
from ml_metrics import aggregates
from ml_metrics import pipeline
from ml_metrics._src.aggregates import stats
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
import numpy as np
import portpicker

Task = courier_worker.Task


# Required for BNS resolution.
def setUpModule():
  testutil.SetupMockBNS()


class CourierWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server_wrapper = courier_server.CourierServerWrapper()
    self.server = self.server_wrapper.build_server()
    self.t = threading.Thread(target=self.server_wrapper.run_until_shutdown)
    self.t.start()
    self.worker = courier_worker.Worker(self.server.address)
    self.worker.wait_until_alive()

  def tearDown(self):
    self.worker.shutdown()
    self.t.join()
    super().tearDown()

  def test_worker_call(self):
    self.assertEqual(
        ['echo'],
        courier_worker.get_results([self.worker.call('echo')], blocking=True),
    )

  def test_worker_run_task(self):
    task = Task.new('echo').add_task(
        Task.new(lazy_fns.trace(len)([1, 2]), blocking=True)
    )
    result = self.worker.run_task(task)
    self.assertEqual(2, lazy_fns.maybe_make(result.state.result()))
    self.assertEqual(
        'echo', lazy_fns.maybe_make(result.parent_task.state.result())
    )

  def test_worker_heartbeat(self):
    # Server is not started, thus it is never alive.
    worker = courier_worker.Worker(
        f'localhost:{portpicker.pick_unused_port()}',
        heartbeat_threshold_secs=0,
        call_timeout=0.01,
    )
    self.assertFalse(worker.is_alive)

  def test_worker_pendings(self):
    self.worker.call(lazy_fns.trace(time.sleep)(0.3))
    self.assertLen(self.worker.pendings, 1)
    # wait until the call is finished.
    time.sleep(0.6)
    self.assertEmpty(self.worker.pendings)

  def test_worker_idle(self):
    self.assertTrue(self.worker.has_capacity)
    self.worker.call(lazy_fns.trace(time.sleep)(0.3))
    self.assertFalse(self.worker.has_capacity)
    # wait until the call is finished.
    time.sleep(0.6)
    self.assertTrue(self.worker.has_capacity)

  def test_worker_exception(self):
    state_futures = [self.worker.call(lazy_fns.trace(len)(0.3))]
    courier_worker.wait_until_done(state_futures)
    exceptions = courier_worker.get_exceptions(state_futures)
    self.assertLen(exceptions, 1)
    self.assertIsInstance(exceptions[0], Exception)

  def test_worker_timeout(self):
    self.worker.set_timeout(0.01)
    state = self.worker.call(lazy_fns.trace(time.sleep)(0.3))
    time.sleep(0.6)
    exceptions = courier_worker.get_exceptions([state])
    self.assertLen(exceptions, 1)
    self.assertIsInstance(exceptions[0], Exception)

  def test_worker_shutdown(self):
    server_wrapper = courier_server.CourierServerWrapper()
    server = server_wrapper.build_server()
    t = threading.Thread(target=server_wrapper.run_until_shutdown)
    t.start()
    worker = courier_worker.Worker(server.address)
    self.assertTrue(worker.call(True))
    self.assertTrue(t.is_alive())
    worker.shutdown()
    time.sleep(3)
    self.assertFalse(t.is_alive())


class CourierWorkerGroupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server_wrapper = courier_server.CourierServerWrapper()
    self.server = self.server_wrapper.build_server()
    self.t = threading.Thread(target=self.server_wrapper.run_until_shutdown)
    self.t.start()
    self.worker_pool = courier_worker.WorkerPool([self.server.address])
    self.worker_pool.wait_until_alive(deadline_secs=12)

  def tearDown(self):
    self.worker_pool.shutdown()
    self.t.join()
    super().tearDown()

  def test_worker_group_call(self):
    actual = self.worker_pool.call_and_wait('echo')
    self.assertEqual(['echo'], actual)

  def test_worker_group_run_and_iterate(self):
    def mock_generator(n):
      yield from range(n)

    tasks = [
        Task.new('echo').add_task(
            lazy_fns.trace(mock_generator)(3), blocking=True
        )
    ] * 3
    courier_worker._LOGGING_INTERVAL_SEC = 0.1
    with self.assertLogs(level='INFO') as cm:
      results = [
          result
          for result in self.worker_pool.run_and_iterate(
              tasks, sleep_interval=0.01, num_total_failures_threshold=0
          )
      ]
    self.assertNotEmpty([l for l in cm.output if 'iterate progress' in l])
    self.assertLen(results, 9)
    self.assertCountEqual(list(range(3)) * 3, results)

  def test_worker_group_run_and_iterate_invalid_iterator(self):
    tasks = [
        Task.new('echo').add_task(lazy_fns.trace(len)([3]), blocking=True)
    ] * 3
    with self.assertRaises(ValueError):
      list(
          self.worker_pool.run_and_iterate(
              tasks, num_total_failures_threshold=0
          )
      )

  def test_worker_group_run_tasks(self):
    tasks = [
        Task.new('echo', blocking=False).add_task(lazy_fns.trace(len)([1, 2]))
    ] * 3
    states = [task.state for task in self.worker_pool.run_tasks(tasks)]
    self.assertLen(states, 3)
    actual = courier_worker.get_results(states)
    self.assertEqual([2] * 3, actual)

  def test_worker_group_idle_workers(self):
    self.assertLen(self.worker_pool.idle_workers(), 1)
    self.worker_pool.idle_workers()[0].call(lazy_fns.trace(time.sleep)(0.3))
    self.assertEmpty(self.worker_pool.idle_workers())

  def test_worker_group_shutdown(self):
    server_wrapper = courier_server.CourierServerWrapper()
    server = server_wrapper.build_server()
    t = threading.Thread(target=server_wrapper.run_until_shutdown)
    t.start()
    worker_group = courier_worker.WorkerPool([server.address])
    self.assertTrue(worker_group.call_and_wait(True))
    worker_group.shutdown()
    time.sleep(3)
    self.assertFalse(t.is_alive())

  def test_worker_group_failed_to_start(self):
    worker_group = courier_worker.WorkerPool(
        [f'localhost:{portpicker.pick_unused_port()}'],
        call_timeout=0.01,
    )
    try:
      with self.assertLogs(level='WARNING') as cm:
        worker_group.wait_until_alive(deadline_secs=12)
      self.assertRegex(cm.output[0], '.*missed a heartbeat.*')
      self.assertRegex(cm.output[1], 'Failed to connect to workers.*')
    except ValueError:
      pass  # The exception is tested below.
    with self.assertRaises(ValueError):
      worker_group.wait_until_alive(deadline_secs=12)

  def test_worker_group_num_workers(self):
    worker_group = courier_worker.WorkerPool(['a', 'b'])
    self.assertEqual(2, worker_group.num_workers)


class CourierWorkerCoordiationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.servers = []
    self.threads = []
    for _ in range(2):
      server_wrapper = courier_server.CourierServerWrapper()
      self.servers.append(server_wrapper.build_server().address)
      self.threads.append(
          threading.Thread(target=server_wrapper.run_until_shutdown)
      )
    for t in self.threads:
      t.start()
    self.worker_pool = courier_worker.WorkerPool(self.servers)
    self.worker_pool.wait_until_alive(deadline_secs=12)

  def tearDown(self):
    _ = [t.join() for t in self.threads]
    super().tearDown()

  def test_coordinate_call(self):

    def random_numbers_iterator(
        shard_index: int,
        num_shards: int,
        total_numbers: int,
        batch_size: int,
    ):
      num_batches = max(total_numbers // batch_size, 1)
      for i in range(num_batches):
        if i % num_shards == shard_index:
          yield np.random.randint(100, size=batch_size)

    def define_pipeline(total_numbers: int, shard_index: int, num_shards: int):
      return (
          pipeline.Pipeline.new()
          .data_source(
              random_numbers_iterator(
                  shard_index,
                  num_shards,
                  total_numbers=total_numbers,
                  batch_size=100,
              )
          )
          .aggregate(
              output_keys='stats',
              fn=aggregates.MergeableMetricAggFn(stats.StatsState()),
          )
      )

    results = self.worker_pool.aggregate_by_pipeline(
        define_pipeline,
        total_numbers=1000,
    )
    self.worker_pool.shutdown()
    self.assertIsNotNone(results)
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], stats.StatsState)


if __name__ == '__main__':
  absltest.main()
