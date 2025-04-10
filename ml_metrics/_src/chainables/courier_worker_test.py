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
import queue
import time

from absl.testing import absltest
from absl.testing import parameterized
from courier.python import testutil
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import courier_utils
from ml_metrics._src.utils import iter_utils
import portpicker


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
  courier_server.shutdown_all()


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


class CourierWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServer(
        'CourierWorker', clients=['CourierWorker']
    )
    self.server.start()
    self.worker = courier_worker.Worker(self.server.address)

  def test_worker_call(self):
    self.assertEqual(
        ['echo'],
        courier_worker.get_results([self.worker.call('echo')]),
    )

  def test_worker_str(self):
    self.assertRegex(
        str(self.worker),
        r'Worker\("CourierWorker", timeout=.+, from_last_heartbeat=.+\)',
    )

  def test_task_done(self):
    task = courier_worker.Task.new('echo')
    self.assertFalse(task.done())
    task = self.worker.submit(task)
    courier_worker.wait([task])
    self.assertTrue(task.done())

  def test_wait_timeout(self):
    task = courier_worker.Task.new(lazy_fns.trace(time.sleep)(1))
    task = self.worker.submit(task)
    self.assertNotEmpty(courier_worker.wait([task], timeout=0).not_done)

  def test_worker_exception(self):
    state_futures = [self.worker.call(lazy_fns.trace(len)(0.3))]
    exceptions = courier_worker.get_exceptions(state_futures)
    self.assertLen(exceptions, 1)
    self.assertIsInstance(exceptions[0], Exception)

  def test_worker_timeout(self):
    self.worker.call_timeout = 0.01
    state = self.worker.call(lazy_fns.trace(time.sleep)(0.3))
    exceptions = courier_worker.get_exceptions([state])
    self.assertLen(exceptions, 1)
    self.assertIsInstance(exceptions[0], Exception)


class TestServer(courier_server.CourierServer):
  """Test server for CourierServer."""

  def set_up(self):
    super().set_up()

    def plus_one(x: int | bytes):
      if isinstance(x, bytes):
        x = lazy_fns.pickler.loads(x)
      return lazy_fns.pickler.dumps(x + 1)

    assert self._server is not None
    self._server.Bind('plus_one', plus_one)


class CourierWorkerPoolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServer(
        'WorkerPool', clients=['WorkerPool']
    )
    self.server.start()
    self.worker = courier_worker.Worker(self.server.address)
    self.worker_pool = courier_worker.WorkerPool([self.worker])
    self.worker_pool.wait_until_alive(deadline_secs=15)

  @parameterized.named_parameters([
      dict(testcase_name='from_workers'),
      dict(
          testcase_name='with_different_timeout',
          workerpool_params=dict(call_timeout=1),
      ),
      dict(
          testcase_name='with_different_max_parallelism',
          workerpool_params=dict(max_parallelism=2),
      ),
      dict(
          testcase_name='with_different_heartbeat_threshold_secs',
          workerpool_params=dict(heartbeat_threshold_secs=2),
      ),
      dict(
          testcase_name='with_different_iterate_batch_size',
          workerpool_params=dict(iterate_batch_size=2),
      ),
  ])
  def test_worker_pool_construct_from_workers(self, workerpool_params=None):
    workerpool_params = workerpool_params or {}
    worker_pool = courier_worker.WorkerPool(
        self.worker_pool.all_workers, **workerpool_params
    )
    assert len(worker_pool.all_workers) == 1
    worker = worker_pool.all_workers[0]
    if workerpool_params:
      self.assertIsNot(worker, self.worker_pool.all_workers[0])
    else:
      self.assertIs(worker, self.worker_pool.all_workers[0])
    for x in workerpool_params:
      self.assertEqual(getattr(worker, x), workerpool_params[x])

  def test_worker_pool_addresses(self):
    self.assertEqual([self.server.address], self.worker_pool.addresses)

  def test_worker_pool_call(self):
    actual = self.worker_pool.call_and_wait('echo')
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertEqual(['echo'], actual)

  def test_worker_pool_call_with_method_in_task(self):
    server = TestServer('test_server', clients=['test_server'])
    server.start()
    worker_pool = courier_worker.WorkerPool([server.address])
    task = courier_utils.Task.new(1, courier_method='plus_one')
    # We only have one task, so just return the first element.
    self.assertEqual(2, worker_pool.run(task))
    worker_pool.shutdown()

  def test_worker_pool_run(self):
    tasks = lazy_fns.trace(len)([1, 2])
    actual = self.worker_pool.run(tasks)
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertEqual(2, actual)

  def test_worker_cache_info(self):
    self.worker_pool.run(lazy_fns.trace(len)((1, 2), cache_result_=True))
    hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.worker_pool.run(lazy_fns.trace(len)((1, 2), cache_result_=True))
    new_hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.assertEqual(new_hits - hits, 1)

  def test_worker_cache_by_id(self):
    # The list is not hashable, this will fall back to hashing by id.
    fn = lazy_fns.trace(len)([1, 2], cache_result_=True)
    self.worker_pool.run(fn)
    hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.worker_pool.run(fn)
    new_hits = self.worker_pool.idle_workers()[0].cache_info().hits
    self.assertEqual(new_hits - hits, 1)

  def test_worker_pool_idle_workers(self):
    worker_pool = courier_worker.WorkerPool([self.server.address])
    worker_pool.wait_until_alive(deadline_secs=15)
    idle_workers = worker_pool.idle_workers()
    self.assertLen(idle_workers, 1)
    idle_workers[0].call(lazy_fns.trace(time.sleep)(1))
    self.assertEmpty(worker_pool.idle_workers())

  def test_worker_pool_shutdown(self):
    server = courier_server.CourierServer()
    t = server.start()
    worker_pool = courier_worker.WorkerPool([server.address])
    worker_pool.wait_until_alive(deadline_secs=15)
    self.assertTrue(worker_pool.call_and_wait(True))
    worker_pool.shutdown()
    ticker = time.time()
    while t.is_alive():
      time.sleep(0)
      if time.time() - ticker > 10:
        self.fail('Server is not shutdown after 10 seconds.')

  def test_worker_pool_fail_to_start(self):
    port = portpicker.pick_unused_port()
    worker_pool = courier_worker.WorkerPool([f'localhost:{port}'])
    with self.assertRaisesRegex(ValueError, 'Failed to connect to minimum.*'):
      worker_pool.wait_until_alive(deadline_secs=1)

  def test_worker_pool_num_workers(self):
    addrs = [
        f'localhost:{portpicker.pick_unused_port()}',
        f'localhost:{portpicker.pick_unused_port()}',
    ]
    worker_pool = courier_worker.WorkerPool(addrs)
    self.assertEqual(2, worker_pool.num_workers)


# TODO: b/356633410 - Remove after deprecating worker_pool.iterate.
class CouierWorkerPoolGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.PrefetchedCourierServer('GeneratorWorker')
    self.server.start()
    self.worker = courier_worker.Worker(self.server.address)
    self.worker_pool = courier_worker.WorkerPool([self.worker])
    self.worker_pool.wait_until_alive(deadline_secs=15)

  def test_worker_pool_iterate_lazy_generator(self):
    lazy_generators = (lazy_fns.trace(mock_generator)(3) for _ in range(30))
    courier_worker._LOGGING_INTERVAL_SEC = 0.01
    generator_result_queue = queue.SimpleQueue()
    # Adds a worker that is not reachable.
    worker_pool = courier_worker.WorkerPool(
        [
            TIMEOUT_SERVER.address,
            UNREACHABLE_SERVER.address,
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
              retry_threshold=0,
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

  def test_worker_pool_iterate_by_task(self):
    tasks = [lazy_fns.trace(mock_generator)(3)] * 5
    courier_worker._LOGGING_INTERVAL_SEC = 0.01
    generator_result_queue = queue.SimpleQueue()
    with self.assertLogs(level='INFO') as cm:
      results = [
          result
          for result in self.worker_pool.iterate(
              tasks,
              generator_result_queue=generator_result_queue,
              retry_threshold=0,
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
  # def test_worker_pool_iterate_invalid_iterator(self):
  #   invalid_iterators = [lazy_fns.trace(len)([3])]
  #   generator_result_queue = queue.SimpleQueue()
  #   iterator = self.worker_pool.iterate(
  #       invalid_iterators,
  #       retry_threshold=0,
  #       generator_result_queue=generator_result_queue,
  #   )
  #   with self.assertRaises(Exception):
  #     next(iterator)


if __name__ == '__main__':
  absltest.main()
