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
import math
import queue
import threading
import time

from absl.testing import absltest
from absl.testing import parameterized
from courier.python import testutil
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import io
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import orchestrate
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import test_utils
import more_itertools as mit
import portpicker


_PIPELINE = transform.TreeTransform


class TimeoutServer(courier_server.CourierServer):
  """Test server for PrefetchedCourierServer."""

  def set_up(self):
    super().set_up()

    def timeout(x):
      time.sleep(10)
      result = lazy_fns.maybe_make(x)
      return lazy_fns.pickler.dumps(result)

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
  courier_server.shutdown_all()


class RunAsCompletedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server.CourierServer('WorkerGroup', clients=['host'])
    self.server.start()
    self.always_timeout_server = TimeoutServer('timeout', clients=['host'])
    self.always_timeout_server.start()
    self.worker_pool = courier_worker.WorkerPool([self.server.address])
    self.unreachable_address = f'localhost:{portpicker.pick_unused_port()}'
    courier_worker.wait_until_alive(self.server.address)
    courier_worker.wait_until_alive(self.always_timeout_server.address)

  def test_run_multi_tasks(self):
    tasks = (lazy_fns.trace(len)([1, 2]) for _ in range(3))
    actual = list(orchestrate.as_completed(self.worker_pool, tasks))
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertLen(actual, 3)
    self.assertEqual([2] * 3, actual)

  def test_as_completed_with_retry(self):
    tasks = [lazy_fns.trace(len)([1, 2]) for _ in range(3)]
    addrs = [
        self.always_timeout_server.address,
        self.unreachable_address,
        self.server.address,
    ]
    worker_pool = courier_worker.WorkerPool(addrs, call_timeout=6)
    # Add a worker with invalid address to test the retry logic.
    self.assertLen(worker_pool.workers, 2)
    actual = list(orchestrate.as_completed(worker_pool, tasks))
    self.assertLen(actual, 3)
    self.assertEqual([2] * 3, actual)

  def test_worker_pool_as_completed_with_exception(self):
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

    tasks = [lazy_fns.trace(blocking_fn)(3)] * 3
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
    tasks = [lazy_fns.trace(len)([1, 2])]
    actual = list(orchestrate.as_completed(self.worker_pool, tasks))
    self.assertEqual([2], actual)
    self.assertEmpty(shared_worker_pool.acquired_workers)


class RunShardedIteratorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.servers = [
        courier_server.PrefetchedCourierServer(
            f'server_sharded_{i}',
            clients=['host'],
            prefetch_size=1024,
            ignore_error=False,
        )
        for i in range(2)
    ]
    for s in self.servers:
      s.start()
    addrs = [s.address for s in self.servers]
    self.worker_pool = courier_worker.WorkerPool(addrs, iterate_batch_size=4096)

  @parameterized.named_parameters([
      dict(
          testcase_name='with_agg_unfused',
          fuse_aggregate=False,
      ),
      dict(
          testcase_name='with_agg_fused',
          fuse_aggregate=True,
      ),
  ])
  def test_in_process(self, fuse_aggregate):
    total_numbers, batch_size = 1_000_001, 1000
    pipeline = test_utils.sharded_pipeline(
        total_numbers=total_numbers,
        batch_size=batch_size,
        fuse_aggregate=fuse_aggregate,
    )
    it = pipeline.make().iterate()
    num_batches = mit.ilen(it)
    self.assertEqual(num_batches, math.ceil(total_numbers / batch_size))
    result = it.agg_result
    self.assertIsNotNone(result)
    self.assertIn('stats', result)
    self.assertIsInstance(result['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(result['stats'].count, total_numbers)

  def test_iterator(self):
    results_queue = queue.SimpleQueue()
    total_numbers, batch_size = 1_000_001, 1000
    num_batches = mit.ilen(
        orchestrate.sharded_pipelines_as_iterator(
            self.worker_pool,
            test_utils.sharded_pipeline,
            total_numbers=total_numbers,
            num_shards=self.worker_pool.num_workers + 2,
            batch_size=batch_size,
            num_threads=1,
            result_queue=results_queue,
            retry_failures=False,
        )
    )
    self.assertEqual(num_batches, math.ceil(total_numbers / batch_size))
    results = results_queue.get().agg_result

    self.assertIsNotNone(results)
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, total_numbers)

  def test_iterator_with_exception(self):
    def sharded_pipeline(shard_index: int, num_shards: int):
      ds = io.SequenceDataSource(test_utils.SequenceWithExc(100, 12)).shard(
          shard_index, num_shards
      )
      return transform.TreeTransform.new(name='datasource').data_source(ds)

    with self.assertRaisesRegex(RuntimeError, r'Failed at \d/6 task.'):
      _ = mit.ilen(
          orchestrate.sharded_pipelines_as_iterator(
              self.worker_pool,
              sharded_pipeline,
              num_shards=self.worker_pool.num_workers + 4,
              retry_failures=True,
          )
      )

  def test_iterator_with_timeout_retry(self):
    def sharded_pipeline(shard_index: int, num_shards: int):
      # TimeoutError can be retried.
      ds = io.SequenceDataSource(
          test_utils.SequenceWithExc(100, 12, error_type=TimeoutError)
      ).shard(shard_index, num_shards)
      return transform.TreeTransform.new(name='datasource').data_source(ds)

    with self.assertRaisesRegex(TimeoutError, r'Too many Timeouts: \d > 1'):
      _ = mit.ilen(
          orchestrate.sharded_pipelines_as_iterator(
              self.worker_pool,
              sharded_pipeline,
              num_shards=self.worker_pool.num_workers + 1,
              retry_failures=True,
              retry_threshold=1,
          )
      )


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
          testcase_name='in_process_fused',
          fuse_aggregate=True,
      ),
      dict(
          testcase_name='with_two_workers',
          with_workers=True,
      ),
      dict(
          testcase_name='with_one_worker_fused',
          with_workers=True,
          fuse_aggregate=True,
          num_workers=1,
      ),
      dict(
          testcase_name='with_two_workers_fused',
          with_workers=True,
          fuse_aggregate=True,
      ),
  ])
  def test_run(self, with_workers=False, fuse_aggregate=False, num_workers=2):
    total_numbers, batch_size = 1_000_001, 1000
    fused_str = '_agg_fused' if fuse_aggregate else ''
    worker_pool = courier_worker.WorkerPool(
        self.worker_pool.all_workers[:num_workers]
    )
    master_server = None
    if with_workers:
      master_server = courier_server.CourierServer(f'master{fused_str}')
    with orchestrate.run_pipeline_interleaved(
        test_utils.sharded_pipeline(
            total_numbers=total_numbers,
            batch_size=batch_size,
            num_threads=1 if with_workers else 0,
            fuse_aggregate=fuse_aggregate,
        ),
        master_server=master_server,
        aggregate_only=fuse_aggregate,
        resources={
            'datasource': orchestrate.RunnerResource(buffer_size=100),
            'apply': orchestrate.RunnerResource(
                worker_pool=worker_pool if with_workers else None,
            ),
        },
    ) as runner:
      it_result = iter(runner.result_queue)
      first_batch = mit.first(it_result)
      cnt = mit.ilen(it_result) + 1
    if with_workers:
      assert master_server is not None
      self.assertTrue(master_server.has_started)
    else:
      self.assertIsNone(master_server)
    if fuse_aggregate:
      self.assertIsNone(first_batch)
    else:
      self.assertIsNotNone(first_batch)
    self.assertEqual(cnt, math.ceil(total_numbers / batch_size))
    self.assertLen(runner.result_queue.returned, 1)
    agg_result = runner.result_queue.returned[0]
    self.assertIsInstance(agg_result, transform.AggregateResult)
    results = agg_result.agg_result
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, total_numbers)

  def test_raises_value_error(self):
    pipeline = (
        _PIPELINE.new(name='datasource')
        .data_source(
            mit.batched(range(5), 2),
        )
        .chain(
            _PIPELINE.new(name='apply')
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
                  timeout=1,
              ),
          },
      ) as runner:
        mit.ilen(runner.result_queue)

if __name__ == '__main__':
  absltest.main()
