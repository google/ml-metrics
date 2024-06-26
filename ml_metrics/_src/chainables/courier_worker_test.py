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
import queue
import time

from absl.testing import absltest
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
import portpicker

Task = courier_worker.Task


# Required for BNS resolution.
def setUpModule():
  testutil.SetupMockBNS()


def mock_generator(n):
  yield from range(n)
  return n


class CourierWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServerWrapper()
    self.server.build_server()
    self.server_thread = self.server.start()
    self.worker = courier_worker.Worker(self.server.address)
    self.worker.wait_until_alive(deadline_secs=12)

  def tearDown(self):
    self.worker.shutdown()
    self.server_thread.join()
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

  def test_worker_async_iterate(self):

    task = Task.new('echo').add_generator_task(
        lazy_fns.trace(mock_generator)(3)
    )
    agg_q = queue.SimpleQueue()
    batch_outputs = []

    async def run():
      async for elem in self.worker.async_iterate(
          task, generator_result_queue=agg_q
      ):
        # During iteration, the worker is considered occupied.
        self.assertFalse(self.worker.has_capacity)
        batch_outputs.append(elem)
      # After iteration, the worker is considered idle.
      self.assertTrue(self.worker.has_capacity)

    asyncio.run(run())
    self.assertEqual(list(range(3)), batch_outputs)
    self.assertEqual(3, agg_q.get())

  def test_worker_async_iterate_raise(self):

    def bad_generator():
      for elem in range(3):
        if elem == 2:
          raise ValueError('bad generator')
        yield elem

    task = courier_worker.GeneratorTask.new(lazy_fns.trace(bad_generator)())
    agg_q = queue.SimpleQueue()

    async def run():
      async for _ in self.worker.async_iterate(
          task, generator_result_queue=agg_q
      ):
        pass

    with self.assertRaises(ValueError):
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
    server = courier_server.CourierServerWrapper()
    server.build_server()
    t = server.start()
    worker = courier_worker.Worker(server.address)
    self.assertTrue(worker.call(True))
    self.assertTrue(t.is_alive())
    worker.shutdown()
    time.sleep(3)
    self.assertFalse(t.is_alive())


class CourierWorkerGroupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServerWrapper()
    self.server.build_server()
    self.server_thread = self.server.start()
    self.worker_pool = courier_worker.WorkerPool([self.server.address])
    self.worker_pool.wait_until_alive(deadline_secs=12)

  def tearDown(self):
    self.worker_pool.shutdown()
    self.server_thread.join()
    super().tearDown()

  def test_worker_group_call(self):
    actual = self.worker_pool.call_and_wait('echo')
    self.assertEqual(['echo'], actual)

  def test_worker_group_run_and_iterate(self):

    tasks = [
        Task.new('echo').add_generator_task(lazy_fns.trace(mock_generator)(3))
    ] * 3
    courier_worker._LOGGING_INTERVAL_SEC = 0.01
    with self.assertLogs(level='INFO') as cm:
      results = [
          result
          for result in self.worker_pool.run_and_iterate(
              tasks, num_total_failures_threshold=0
          )
      ]
    self.assertLen(results, 12)
    self.assertCountEqual([0, 1, 2, 3] * 3, results)
    self.assertNotEmpty([l for l in cm.output if 'progress' in l])

  def test_worker_group_iterate(self):
    tasks = [
        Task.new('echo').add_generator_task(lazy_fns.trace(mock_generator)(3))
    ] * 5
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
    self.assertLen(results, 3 * 5)
    self.assertCountEqual(list(range(3)) * 5, results)
    self.assertEqual(6, generator_result_queue.qsize())
    actual_agg = []
    while not generator_result_queue.empty():
      actual_agg.append(generator_result_queue.get())
    self.assertEqual([3] * 5, actual_agg[:-1])
    self.assertNotEmpty([l for l in cm.output if 'progress' in l])

  def test_worker_group_run_and_iterate_invalid_iterator(self):
    tasks = [Task.new('echo').add_generator_task(lazy_fns.trace(len)([3]))] * 3
    with self.assertRaises(TypeError):
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
    worker_pool = courier_worker.WorkerPool([self.server.address])
    worker_pool.wait_until_alive(deadline_secs=12)
    idle_workers = worker_pool.idle_workers()
    self.assertLen(idle_workers, 1)
    idle_workers[0].call(lazy_fns.trace(time.sleep)(3))
    self.assertEmpty(worker_pool.idle_workers())

  def test_worker_group_shutdown(self):
    server = courier_server.CourierServerWrapper()
    server.build_server()
    t = server.start()
    worker_group = courier_worker.WorkerPool([server.address])
    worker_group.wait_until_alive(deadline_secs=12)
    self.assertTrue(worker_group.call_and_wait(True))
    worker_group.shutdown()
    time.sleep(3)
    self.assertFalse(t.is_alive())

  def test_worker_group_failed_to_start(self):
    worker_group = courier_worker.WorkerPool(
        [f'localhost:{portpicker.pick_unused_port()}'],
        call_timeout=0.01,
        heartbeat_threshold_secs=3,
    )
    try:
      with self.assertLogs(level='WARNING') as cm:
        worker_group.wait_until_alive(deadline_secs=6)
      self.assertRegex(cm.output[0], '.*missed a heartbeat.*')
      self.assertRegex(cm.output[1], 'Failed to connect to workers.*')
    except ValueError:
      pass  # The exception is tested below.
    with self.assertRaises(ValueError):
      worker_group.wait_until_alive(deadline_secs=6)

  def test_worker_group_num_workers(self):
    worker_group = courier_worker.WorkerPool(['a', 'b'])
    self.assertEqual(2, worker_group.num_workers)


if __name__ == '__main__':
  absltest.main()
