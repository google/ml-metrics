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

from absl.testing import absltest
from courier.python import testutil
from ml_metrics import aggregates
from ml_metrics import chainable
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.chainables import orchestrate
import more_itertools as mit
import numpy as np


SERVER_ADDRS = [f'server_{i}' for i in range(2)]


def setUpModule():
  # Required for BNS resolution.
  testutil.SetupMockBNS()
  _ = [courier_server._cached_server(addr) for addr in SERVER_ADDRS]


def tearDownModule():
  courier_worker.WorkerPool(SERVER_ADDRS).shutdown()


def random_numbers_iterator(
    total_numbers: int,
    batch_size: int,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
):
  num_batches, residual = divmod(total_numbers, batch_size)
  for i in range(num_batches):
    if i % num_shards == shard_index:
      yield np.random.randint(100, size=batch_size)
  if residual:
    yield np.random.randint(100, size=residual)


class OrchestrateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.worker_pool = courier_worker.WorkerPool(SERVER_ADDRS)
    self.worker_pool.wait_until_alive(deadline_secs=12)

  def test_sharded_pipelines_as_iterator(self):

    def define_pipeline(shard_index: int, num_shards: int):
      return (
          chainable.Pipeline.new()
          .data_source(
              random_numbers_iterator(
                  total_numbers=1000,
                  batch_size=100,
                  shard_index=shard_index,
                  num_shards=num_shards,
              )
          )
          .aggregate(
              output_keys='stats',
              fn=aggregates.MergeableMetricAggFn(
                  rolling_stats.MeanAndVariance()
              ),
          )
      )

    results_queue = queue.SimpleQueue()
    for elem in orchestrate.sharded_pipelines_as_iterator(
        self.worker_pool,
        define_pipeline,
        result_queue=results_queue,
        retry_failures=False,
    ):
      self.assertEqual(elem.size, 100)

    results = results_queue.get().agg_result

    self.assertIsNotNone(results)
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, 1000)

  def test_run_pipelines_interleaved_default(self):
    pipeline = (
        chainable.Pipeline.new(name='datasource')
        .data_source(
            mit.batched(range(1001), 32),
        )
        .chain(
            chainable.Pipeline.new(name='apply')
            .apply(fn=np.asarray, output_keys='inputs')
            .assign('feature1', fn=lambda x: x + 1, input_keys='inputs')
        )
        .chain(
            chainable.Pipeline.new(name='agg').aggregate(
                output_keys='stats',
                fn=aggregates.MergeableMetricAggFn(
                    rolling_stats.MeanAndVariance()
                ),
                input_keys='feature1',
            )
        )
    )
    runner_state = orchestrate.run_pipeline_interleaved(
        pipeline,
        master_server=courier_server.CourierServerWrapper('master_interleaved'),
        ignore_failures=False,
        resources={
            'datasource': orchestrate.RunnerResource(buffer_size=6),
            'apply': orchestrate.RunnerResource(
                worker_pool=self.worker_pool,
                buffer_size=6,
            ),
            'agg': orchestrate.RunnerResource(timeout=15),
        },
    )
    result_queue = runner_state.result_queue
    cnt = mit.ilen(result_queue.dequeue_as_iterator())
    runner_state.wait_and_maybe_raise()
    self.assertEqual(cnt, 32)  # ceil(1001/32) = 32 batches.
    self.assertLen(result_queue.returned, 1)
    agg_result = result_queue.returned[0]
    self.assertIsInstance(agg_result, chainable.AggregateResult)
    results = agg_result.agg_result
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, 1001)

  def test_run_pipelines_interleaved_raises(self):
    pipeline = (
        chainable.Pipeline.new()
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
      runner_state = orchestrate.run_pipeline_interleaved(
          pipeline,
          master_server=master_server,
          ignore_failures=False,
          resources={
              'apply': orchestrate.RunnerResource(
                  worker_pool=self.worker_pool,
                  buffer_size=1,
              ),
          },
      )
      runner_state.wait_and_maybe_raise()


if __name__ == '__main__':
  absltest.main()
