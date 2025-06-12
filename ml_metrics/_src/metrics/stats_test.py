# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for stats."""

from absl.testing import parameterized
from ml_metrics._src.metrics import stats
import numpy as np

from absl.testing import absltest

_BATCH_WITH_NAN = np.concatenate(
    [np.random.randn(1000) + 1e7, [np.nan] * 100], axis=0
)


class StatsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='variance',
          metric_fn=stats.var,
          expected=np.nanvar(_BATCH_WITH_NAN),
      ),
      dict(
          testcase_name='stddev',
          metric_fn=stats.stddev,
          expected=np.nanstd(_BATCH_WITH_NAN),
      ),
      dict(
          testcase_name='mean',
          metric_fn=stats.mean,
          expected=np.nanmean(_BATCH_WITH_NAN),
      ),
      dict(
          testcase_name='count',
          metric_fn=stats.count,
          expected=np.nansum(~np.isnan(_BATCH_WITH_NAN)),
      ),
      dict(
          testcase_name='total',
          metric_fn=stats.total,
          expected=np.nansum(_BATCH_WITH_NAN),
      ),
  ])
  def test_stats_individual_metrics(self, metric_fn, expected):
    got = metric_fn(_BATCH_WITH_NAN)
    np.testing.assert_almost_equal(got, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='variance',
          metric_fn=stats.var,
          expected=np.nan,
      ),
      dict(
          testcase_name='stddev',
          metric_fn=stats.stddev,
          expected=np.nan,
      ),
      dict(
          testcase_name='mean',
          metric_fn=stats.mean,
          expected=np.nan,
      ),
      dict(
          testcase_name='count',
          metric_fn=stats.count,
          expected=0,
      ),
      dict(
          testcase_name='total',
          metric_fn=stats.total,
          expected=0.0,
      ),
  ])
  def test_stats_individual_metrics_empty_batch(self, metric_fn, expected):
    got = metric_fn([])
    np.testing.assert_almost_equal(got, expected)


if __name__ == '__main__':
  absltest.main()
