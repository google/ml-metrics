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
from ml_metrics import aggregates
from ml_metrics import chainable
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import orchestrate
import more_itertools as mit
import numpy as np
import portpicker

# For test, accelerate the heartbeat interval.
courier_worker._HRTBT_INTERVAL_SECS = 0.1
courier_worker._HRTBT_THRESHOLD_SECS = 1


class TimeoutServer(courier_server.CourierServerWrapper):
  """Test server for CourierServerWrapper."""

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


SERVER_ADDRS = [f'server_{i}' for i in range(2)]
ALWAYS_TIMEOUT_SERVER = TimeoutServer()


def setUpModule():
  # Required for BNS resolution.
  testutil.SetupMockBNS()
  courier_server._cached_server('WorkerGroup')
  for addr in SERVER_ADDRS:
    courier_server._cached_server(addr)
  ALWAYS_TIMEOUT_SERVER.start(daemon=True)


def tearDownModule():
  threads = [courier_server._cached_server('WorkerGroup').stop()]
  for addr in SERVER_ADDRS:
    threads.append(courier_server._cached_server(addr).stop())
  threads.append(ALWAYS_TIMEOUT_SERVER.stop())
  _ = [t.join() for t in threads if t]


def sharded_ones(
    total_numbers: int,
    batch_size: int,
    shard_index: int = 0,
    num_shards: int = 1,
):
  num_batches, remainder = divmod(total_numbers, batch_size)
  for i in range(num_batches):
    if i % num_shards == shard_index:
      yield batch_size
  if not shard_index and remainder:
    yield remainder


def sharded_pipeline(
    total_numbers: int,
    batch_size: int,
    shard_index: int = 0,
    num_shards: int = 1,
    num_threads: int = 0,
):
  return (
      chainable.Pipeline.new(name='datasource')
      .data_source(
          sharded_ones(
              total_numbers,
              batch_size=batch_size,
              shard_index=shard_index,
              num_shards=num_shards,
          )
      )
      .chain(
          chainable.Pipeline.new(name='apply', num_threads=num_threads).apply(
              fn=lambda batch_size: np.random.randint(100, size=batch_size),
          )
      )
      .chain(
          chainable.Pipeline.new(name='agg').aggregate(
              output_keys='stats',
              fn=aggregates.MergeableMetricAggFn(
                  rolling_stats.MeanAndVariance()
              ),
          )
      )
  )


class RunAsCompletedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.server = courier_server._cached_server('WorkerGroup')
    self.worker_pool = courier_worker.WorkerPool([self.server.address])
    self.unreachable_address = f'localhost:{portpicker.pick_unused_port()}'
    self.worker_pool.wait_until_alive(deadline_secs=12)

  def test_run_multi_tasks(self):
    tasks = (chainable.trace(len)([1, 2]) for _ in range(3))
    actual = list(orchestrate.as_completed(self.worker_pool, tasks))
    self.assertEmpty(self.worker_pool.acquired_workers)
    self.assertLen(actual, 3)
    self.assertEqual([2] * 3, actual)

  def test_as_completed_with_retry(self):
    tasks = [chainable.trace(len)([1, 2]) for _ in range(30)]
    addrs = [
        ALWAYS_TIMEOUT_SERVER.address,
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
    actual = list(orchestrate.as_completed(worker_pool, tasks))
    self.assertLen(actual, 30)
    self.assertEqual([2] * 30, actual)

  def test_worker_pool_as_completed_with_exception(self):
    def foo():
      raise ValueError('foo')

    tasks = [chainable.trace(foo)() for _ in range(3)]
    addrs = [
        ALWAYS_TIMEOUT_SERVER.address,
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
      next(orchestrate.as_completed(worker_pool, tasks))
    self.assertEmpty(
        list(orchestrate.as_completed(worker_pool, tasks, ignore_failures=True))
    )

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
    time.sleep(0)
    tasks = [chainable.trace(len)([1, 2])]
    actual = list(orchestrate.as_completed(self.worker_pool, tasks))
    self.assertEqual([2], actual)
    self.assertEmpty(shared_worker_pool.acquired_workers)


class RunShardedIteratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.worker_pool = courier_worker.WorkerPool(SERVER_ADDRS, call_timeout=6)
    self.worker_pool.wait_until_alive(deadline_secs=12)

  def test_iterator(self):
    results_queue = queue.SimpleQueue()
    for elem in orchestrate.sharded_pipelines_as_iterator(
        self.worker_pool,
        sharded_pipeline,
        total_numbers=1000,
        batch_size=100,
        result_queue=results_queue,
        retry_failures=False,
    ):
      self.assertEqual(elem.size, 100)

    results = results_queue.get().agg_result

    self.assertIsNotNone(results)
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, 1000)


class RunInterleavedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.worker_pool = courier_worker.WorkerPool(SERVER_ADDRS, call_timeout=12)
    self.worker_pool.wait_until_alive(deadline_secs=12)

  @parameterized.named_parameters([
      dict(
          testcase_name='in_process',
      ),
      dict(
          testcase_name='with_worker_on_apply',
          with_worker_on_apply=True,
      ),
      dict(
          testcase_name='with_worker_on_agg',
          with_worker_on_agg=True,
      ),
  ])
  def test_default_config(
      self, with_worker_on_apply=False, with_worker_on_agg=False
  ):
    total_examples = 1001
    with orchestrate.run_pipeline_interleaved(
        sharded_pipeline(
            total_numbers=total_examples, batch_size=32, num_threads=4
        ),
        master_server=courier_server.CourierServerWrapper(),
        ignore_failures=False,
        resources={
            'datasource': orchestrate.RunnerResource(buffer_size=6),
            'apply': orchestrate.RunnerResource(
                worker_pool=self.worker_pool if with_worker_on_apply else None,
                buffer_size=6,
            ),
            'agg': orchestrate.RunnerResource(
                worker_pool=self.worker_pool if with_worker_on_agg else None,
                timeout=30,
            ),
        },
    ) as runner:
      result_queue = runner.result_queue
      cnt = mit.ilen(result_queue)
    self.assertEqual(cnt, int(total_examples / 32) + 1)
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
    with self.assertRaises(ValueError):
      master_server = courier_server.CourierServerWrapper('master_raises')
      with orchestrate.run_pipeline_interleaved(
          pipeline,
          master_server=master_server,
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
