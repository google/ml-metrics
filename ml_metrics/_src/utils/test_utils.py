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
"""Utilitities for testing, internal use only."""

from ml_metrics import aggregates
from ml_metrics import chainable
from ml_metrics.metrics import rolling_stats
import numpy as np


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
    fuse_aggregate: bool = True,
    num_threads: int = 0,
):
  """A pipeline to calculate the stats of batches of random integers."""
  data_pipeline = chainable.Pipeline.new(name='datasource').data_source(
      sharded_ones(
          total_numbers,
          batch_size=batch_size,
          shard_index=shard_index,
          num_shards=num_shards,
      )
  )
  pipeline = data_pipeline.chain(
      chainable.Pipeline.new(name='apply', num_threads=num_threads).apply(
          fn=lambda batch_size: np.random.randint(100, size=batch_size),
      )
  )

  if fuse_aggregate:
    return pipeline.aggregate(
        output_keys='stats',
        fn=aggregates.MergeableMetricAggFn(rolling_stats.MeanAndVariance()),
    )
  return pipeline.chain(
      chainable.Pipeline.new(name='agg').aggregate(
          output_keys='stats',
          fn=aggregates.MergeableMetricAggFn(rolling_stats.MeanAndVariance()),
      )
  )
