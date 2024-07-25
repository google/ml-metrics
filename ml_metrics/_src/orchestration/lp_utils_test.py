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
"""Launchpad integration test for the consumer_producers example."""

from absl.testing import absltest
from absl.testing import parameterized
import launchpad as lp
from ml_metrics import aggregates
from ml_metrics import pipeline
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.orchestration import lp_utils
import numpy as np


def random_numbers_iterator(
    shard_index: int,
    num_shards: int,
    total_numbers: int,
    batch_size: int = 1000,
):
  num_batches = max(1, total_numbers // batch_size)
  for i in range(num_batches):
    if i % num_shards == shard_index:
      yield np.random.randint(100, size=batch_size)


def sharded_pipeline(total_numbers: int, shard_index: int, num_shards: int):
  return (
      pipeline.Pipeline.new()
      .data_source(
          random_numbers_iterator(
              shard_index,
              num_shards,
              total_numbers=total_numbers,
              batch_size=1_000_000,
          )
      )
      .aggregate(
          output_keys='stats',
          fn=aggregates.MergeableMetricAggFn(rolling_stats.MeanAndVariance()),
      )
  )


class LaunchTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(testcase_name='default', num_workers=0),
      dict(testcase_name='with_remote_workers', num_workers=2),
  ])
  def test_remote_worker_run(self, num_workers: int):
    """Runs the program and makes sure the consumer can run 10 steps."""
    program = lp_utils.build_program_from_pipeline(
        'test_program',
        sharded_pipeline,
        num_workers=num_workers,
        total_numbers=int(6 * 1e6),
        retry_failures=False,
    )

    # Launch all workers declared by the program. Remember to set the launch
    # type here (test & multithreaded).
    try:
      lp.launch(program, launch_type='test_mt', test_case=self)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.fail(f'Failed to run the program: with exception: {e}')


if __name__ == '__main__':
  absltest.main()
