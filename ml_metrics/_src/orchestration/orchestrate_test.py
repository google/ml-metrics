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
"""test for orchestrate."""

import queue

from absl.testing import absltest
from ml_metrics import aggregates
from ml_metrics import pipeline
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.chainables import courier_server
from ml_metrics._src.chainables import courier_worker
from ml_metrics._src.orchestration import orchestrate
import numpy as np


class OrchestrateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.threads = []
    self.servers = []
    for _ in range(2):
      server_wrapper = courier_server.CourierServerWrapper()
      self.servers.append(server_wrapper.build_server().address)
      self.threads.append(server_wrapper.start())
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
              fn=aggregates.MergeableMetricAggFn(
                  rolling_stats.MeanAndVariance()
              ),
          )
      )

    results_queue = queue.SimpleQueue()
    for elem in orchestrate.workerpool_generator(
        self.worker_pool,
        define_pipeline,
        total_numbers=1000,
        result_queue=results_queue,
    ):
      self.assertEqual(elem.size, 100)

    results = results_queue.get().agg_result

    self.worker_pool.shutdown()
    self.assertIsNotNone(results)
    self.assertIn('stats', results)
    self.assertIsInstance(results['stats'], rolling_stats.MeanAndVariance)
    self.assertEqual(results['stats'].count, 1000)


if __name__ == '__main__':
  absltest.main()
