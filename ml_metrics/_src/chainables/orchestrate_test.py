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
import threading
import time

from absl.testing import absltest
from absl.testing import parameterized
from courier.python import testutil
from ml_metrics import chainable
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import orchestrate
from ml_metrics._src.utils import test_utils
import more_itertools as mit
import portpicker


class TimeoutServer(courier_server.CourierServer):
  """Test server for PrefetchedCourierServer."""

  def set_up(self):
    super().set_up()

    def timeout(x):
      time.sleep(10)
      result = chainable.maybe_make(x)
      return chainable.pickler.dumps(result)

    def timeout_init_iterator(x):
      del x
      time.sleep(10)

    assert self._server is not None
    self._server.Unbind('maybe_make')
    self._server.Bind('maybe_make', timeout)
    self._server.Unbind('init_iterator')
    self._server.Bind('init_iterator', timeout_init_iterator)


HOST = courier_server.CourierServer('host')


def setUpModule():
  # Required for BNS resolution.
  testutil.SetupMockBNS()
  HOST.start()


def tearDownModule():
  courier_server.shutdown_all(except_for=['host'])
  HOST.stop().join()


class RunAsCompletedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServer('WorkerGroup', clients=['host'])
    self.server.start()
    self.alswyas_timeout_server = TimeoutServer('timeout', clients=['host'])
    self.alswyas_timeout_server.start()
    self.worker_pool = courier_worker.WorkerPool([self.server.address])
    self.unreachable_address = f'localhost:{portpicker.pick_unused_port()}'
    courier_worker.wait_until_alive(self.server.address)
    courier_worker.wait_until_alive(self.alswyas_timeout_server.address)

  def test_run_multi_tasks(self):
    tasks = (chainable.trace(len)([1, 2]) for _ in range(3))
    actual = list(orchestrate.as_completed(self.worker_pool, tasks))
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertLen(actual, 3)
    self.assertEqual([2] * 3, actual)

  def test_as_completed_with_retry(self):
    tasks = [chainable.trace(len)([1, 2]) for _ in range(30)]
    addrs = [
        self.alswyas_timeout_server.address,
        self.unreachable_address,
        self.server.address,
    ]
    worker_pool = courier_worker.WorkerPool(addrs)
    # Add a worker with invalid address to test the retry logic.
    self.assertLen(worker_pool.workers, 2)
    actual = list(orchestrate.as_completed(worker_pool, tasks))
    self.assertLen(actual, 30)
    self.assertEqual([2] * 30, actual)

  def test_worker_pool_as_completed_with_exception(self):
    def foo():
      raise ValueError('foo')

    tasks = [chainable.trace(foo)() for _ in range(3)]
    addrs = [
        self.alswyas_timeout_server.address,
        self.unreachable_address,
        self.server.address,
    ]
    worker_pool = courier_worker.WorkerPool(
        addrs,
        max_parallelism=1,
        call_timeout=1,
    )
    # Add a worker with invalid address to test the retry logic.
    # Only one worker is alive.
    self.assertLen(worker_pool.workers, 2)
    with self.assertRaisesRegex(Exception, 'foo'):
      next(orchestrate.as_completed(worker_pool, tasks))
    self.assertEmpty(
        list(orchestrate.as_completed(worker_pool, tasks, ignore_failures=True))
    )

  def test_shared_worker_pool_run(self):
    shared_worker_pool = courier_worker.WorkerPool(self.worker_pool.all_workers)
    self.assertNotEmpty(shared_worker_pool.workers)
    self.assertLen(shared_worker_pool.workers, 1)
    self.assertIs(shared_worker_pool.workers[0], self.worker_pool.workers[0])
    blocked = [True]

    def blocking_fn(n):
      cnt = 0
      while blocked[0] and cnt < n:
        time.sleep(0.01)
        cnt += 1

    tasks = [chainable.trace(blocking_fn)(3)] * 3
    t = threading.Thread(
        target=list, args=(orchestrate.as_completed(shared_worker_pool, tasks),)
    )
    t.start()
    while not all(w.is_locked() for w in self.worker_pool.all_workers):
      time.sleep(0)
    # The worker is not acquirable while blocked.
    self.assertSameElements([], self.worker_pool._acquire_all())
    blocked[0] = False
    t.join()
    tasks = [chainable.trace(len)([1, 2])]
    actual = list(orchestrate.as_completed(self.worker_pool, tasks))
    self.assertEqual([2], actual)
    self.assertEmpty(shared_worker_pool.acquired_workers)


class RunShardedIteratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.servers = [
        courier_server.PrefetchedCourierServer(f'server_{i}', clients=['host'])
        for i in range(2)
    ]
    for s in self.servers:
      s.start()
    addrs = [s.address for s in self.servers]
    self.worker_pool = courier_worker.WorkerPool(addrs)

  def test_iterator(self):
    results_queue = queue.SimpleQueue()
    total_numbers, batch_size = 10_001, 100
    num_batches = mit.ilen(
        orchestrate.sharded_pipelines_as_iterator(
            self.worker_pool,
            test_utils.sharded_pipeline,
            total_numbers=total_numbers,
            batch_size=batch_size,
            num_threads=1,
            result_queue=results_queue,
            retry_failures=False,
        )
    )
    self.assertEqual(num_batches, int(total_numbers / batch_size) + 1)
    results = results_queue.get().agg_result

    self.assertIsNotNone(results)
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, total_numbers)


class RunInterleavedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.servers = [
        courier_server.PrefetchedCourierServer(f'server_{i}', clients=['host'])
        for i in range(2)
    ]
    for s in self.servers:
      s.start()
    addrs = [s.address for s in self.servers]
    self.worker_pool = courier_worker.WorkerPool(addrs)

  @parameterized.named_parameters([
      dict(
          testcase_name='in_process',
      ),
      dict(
          testcase_name='with_worker_on_apply',
          with_workers=True,
      ),
      dict(
          testcase_name='with_worker_on_agg',
          with_workers=True,
          fuse_aggregate=True,
      ),
  ])
  def test_default_config(self, with_workers=False, fuse_aggregate=False):
    total_examples = 10_001
    batch_size = 100
    with orchestrate.run_pipeline_interleaved(
        test_utils.sharded_pipeline(
            total_numbers=total_examples,
            batch_size=batch_size,
            num_threads=1,
            fuse_aggregate=fuse_aggregate,
        ),
        master_server=courier_server.CourierServer('master'),
        ignore_failures=False,
        resources={
            'datasource': orchestrate.RunnerResource(buffer_size=16),
            'apply': orchestrate.RunnerResource(
                worker_pool=self.worker_pool if with_workers else None,
                # buffer_size=6,
            ),
        },
    ) as runner:
      result_queue = runner.result_queue
      cnt = mit.ilen(result_queue)
    self.assertEqual(cnt, int(total_examples / batch_size) + 1)
    self.assertLen(result_queue.returned, 1)
    agg_result = result_queue.returned[0]
    self.assertIsInstance(agg_result, chainable.AggregateResult)
    results = agg_result.agg_result
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, total_examples)

  def test_raises_value_error(self):
    pipeline = (
        chainable.Pipeline.new(name='datasource')
        .data_source(
            mit.batched(range(5), 2),
        )
        .chain(
            chainable.Pipeline.new(name='apply')
            # Cannot directly assign the items to a non-dict inputs.
            .assign('feature1', fn=lambda x: x + 1)
        )
    )
    with self.assertRaisesRegex(ValueError, 'stage .* failed'):
      with orchestrate.run_pipeline_interleaved(
          pipeline,
          master_server=courier_server.CourierServer('master_raises'),
          ignore_failures=False,
          resources={
              'apply': orchestrate.RunnerResource(
                  worker_pool=self.worker_pool,
                  buffer_size=1,
              ),
          },
      ) as runner:
        mit.ilen(runner.result_queue)

if __name__ == '__main__':
  absltest.main()
