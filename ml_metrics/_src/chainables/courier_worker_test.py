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
import threading
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
      return lazy_fns.picklers.default.dumps(result)

    def timeout_init_generator(x):
      del x
      time.sleep(60)

    self._server.Unbind('maybe_make')
    self._server.Bind('maybe_make', timeout)
    self._server.Unbind('init_generator')
    self._server.Bind('init_generator', timeout_init_generator)


class CourierWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServerWrapper()
    self.server.build_server()
    self.server_thread = self.server.start(daemon=True)
    self.worker = courier_worker.Worker(self.server.address)
    self.worker.wait_until_alive(deadline_secs=6, sleep_interval_secs=0.1)

  def tearDown(self):
    self.worker.shutdown()
    self.server_thread.join()
    super().tearDown()

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

  def test_worker_run_task(self):
    task = Task.new('echo').add_task(
        Task.new(lazy_fns.trace(len)([1, 2]), blocking=True)
    )
    result = self.worker.submit(task)
    self.assertEqual(2, result.result())
    self.assertEqual(
        'echo', lazy_fns.maybe_make(result.parent_task.state.result())
    )

  def test_wait_timeout(self):
    task = Task.new(lazy_fns.trace(time.sleep)(0.1))
    task = self.worker.submit(task)
    self.assertNotEmpty(courier_worker.wait([task], timeout=0).not_done)

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
        x = lazy_fns.picklers.default.loads(x)
      return lazy_fns.picklers.default.dumps(x + 1)

    self._server.Bind('plus_one', plus_one)


class CourierWorkerGroupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServerWrapper()
    self.server.build_server()
    self.server_thread = self.server.start(daemon=True)
    self.always_timeout_server = TimeoutServer()
    self.always_timeout_server.build_server()
    self.invalid_server_thread = self.always_timeout_server.start(daemon=True)
    self.worker_pool = courier_worker.WorkerPool([self.server.address])
    self.unreachable_address = f'localhost:{portpicker.pick_unused_port()}'
    self.worker_pool.wait_until_alive(deadline_secs=12)

  def tearDown(self):
    self.worker_pool.shutdown()
    self.server_thread.join()
    super().tearDown()

  def test_worker_group_call(self):
    actual = self.worker_pool.call_and_wait('echo')
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertEqual(['echo'], actual)

  def test_worker_group_call_with_method_in_task(self):
    server = TestServer()
    server.build_server()
    thread = server.start(daemon=True)
    tasks = [courier_worker.Task.new(1, courier_method='plus_one')]
    worker_pool = courier_worker.WorkerPool([server.address])
    worker_pool.wait_until_alive(deadline_secs=12)
    states = list(worker_pool.as_completed(tasks))
    # We only have one task, so just return the first element.
    self.assertEqual(2, courier_worker.get_results(states)[0])
    worker_pool.shutdown()
    thread.join()

  def test_worker_group_iterate_lazy_generator(self):
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

  def test_worker_group_iterate_by_task(self):
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
  # def test_worker_group_iterate_invalid_iterator(self):
  #   invalid_iterators = [lazy_fns.trace(len)([3])]
  #   generator_result_queue = queue.SimpleQueue()
  #   iterator = self.worker_pool.iterate(
  #       invalid_iterators,
  #       num_total_failures_threshold=0,
  #       generator_result_queue=generator_result_queue,
  #   )
  #   with self.assertRaises(Exception):
  #     next(iterator)

  def test_shared_worker_pool_run(self):
    shared_worker_pool = courier_worker.WorkerPool(
        self.worker_pool.all_workers, call_timeout=6
    )
    shared_worker_pool.wait_until_alive(deadline_secs=12)
    self.assertNotEmpty(shared_worker_pool.workers)
    blocked = [True]

    def blocking_fn(n):
      cnt = 0
      while blocked[0] and cnt < n:
        time.sleep(0.01)
        cnt += 1

    tasks = [lazy_fns.trace(blocking_fn)(3)] * 3
    t = threading.Thread(target=shared_worker_pool.run, args=(tasks,))
    t.start()
    time.sleep(0)
    # The worker is not acquirable while blocked.
    self.assertSameElements([False], self.worker_pool._acquire_all())
    blocked[0] = False
    time.sleep(0)
    self.assertEqual(2, self.worker_pool.run(lazy_fns.trace(len)([1, 2])))
    self.assertEmpty(shared_worker_pool.acquired_workers)

  def test_worker_group_run(self):
    tasks = lazy_fns.trace(len)([1, 2])
    actual = self.worker_pool.run(tasks)
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertEqual(2, actual)

  def test_worker_group_run_multi_tasks(self):
    tasks = (lazy_fns.trace(len)([1, 2]) for _ in range(3))
    actual = self.worker_pool.run(tasks)
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertLen(actual, 3)
    self.assertEqual([2] * 3, actual)

  def test_worker_cache_info(self):
    self.worker_pool.run(lazy_fns.trace(len, use_cache=True)([1, 2]))
    hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.worker_pool.run(lazy_fns.trace(len, use_cache=True)([1, 2]))
    new_hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.assertEqual(new_hits - hits, 1)

  def test_worker_group_as_completed_with_retry(self):
    tasks = [lazy_fns.trace(len)([1, 2]) for _ in range(30)]
    addrs = [
        self.always_timeout_server.address,
        self.unreachable_address,
        self.server.address,
    ]
    worker_pool = courier_worker.WorkerPool(
        addrs,
        max_parallelism=1,
        call_timeout=1,
    )
    # Add a worker with invalid address to test the retry logic.
    worker_pool.wait_until_alive(deadline_secs=12, minimum_num_workers=2)
    # Only one worker is alive.
    self.assertLen(worker_pool.workers, 2)
    states = list(worker_pool.as_completed(tasks))
    self.assertLen(states, 30)
    actual = courier_worker.get_results(states)
    self.assertEqual([2] * 30, actual)

  def test_worker_group_as_completed_with_exception(self):
    def foo():
      raise ValueError('foo')

    tasks = [lazy_fns.trace(foo)() for _ in range(3)]
    addrs = [
        self.always_timeout_server.address,
        self.unreachable_address,
        self.server.address,
    ]
    worker_pool = courier_worker.WorkerPool(
        addrs,
        max_parallelism=1,
        call_timeout=1,
    )
    # Add a worker with invalid address to test the retry logic.
    worker_pool.wait_until_alive(deadline_secs=12, minimum_num_workers=2)
    # Only one worker is alive.
    self.assertLen(worker_pool.workers, 2)
    with self.assertRaisesRegex(Exception, 'foo'):
      next(worker_pool.as_completed(tasks))
    self.assertEmpty(
        list(worker_pool.as_completed(tasks, ignore_failures=True))
    )

  def test_worker_group_idle_workers(self):
    worker_pool = courier_worker.WorkerPool([self.server.address])
    worker_pool.wait_until_alive(deadline_secs=12, sleep_interval_secs=0)
    idle_workers = worker_pool.idle_workers()
    self.assertLen(idle_workers, 1)
    idle_workers[0].call(lazy_fns.trace(time.sleep)(1))
    self.assertEmpty(worker_pool.idle_workers())

  def test_worker_group_shutdown(self):
    server = courier_server.CourierServerWrapper()
    server.build_server()
    t = server.start()
    worker_group = courier_worker.WorkerPool([server.address])
    worker_group.wait_until_alive(deadline_secs=6, sleep_interval_secs=0.1)
    self.assertTrue(worker_group.call_and_wait(True))
    worker_group.shutdown()
    ticker = time.time()
    while t.is_alive():
      time.sleep(0)
      if time.time() - ticker > 10:
        self.fail('Server is not shutdown after 10 seconds.')

  def test_worker_group_failed_to_start(self):
    worker_group = courier_worker.WorkerPool(
        [f'localhost:{portpicker.pick_unused_port()}'],
        call_timeout=0.01,
        heartbeat_threshold_secs=3,
    )
    try:
      with self.assertLogs(level='WARNING') as cm:
        worker_group.wait_until_alive(deadline_secs=1, sleep_interval_secs=0.1)
      self.assertRegex(cm.output[0], '.*missed a heartbeat.*')
      self.assertRegex(cm.output[1], 'Failed to connect to workers.*')
    except ValueError:
      pass  # The exception is tested below.
    with self.assertRaises(ValueError):
      worker_group.wait_until_alive(deadline_secs=1)

  def test_worker_group_num_workers(self):
    worker_group = courier_worker.WorkerPool(['a', 'b'])
    self.assertEqual(2, worker_group.num_workers)


if __name__ == '__main__':
  absltest.main()
