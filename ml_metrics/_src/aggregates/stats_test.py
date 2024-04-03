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

from ml_metrics._src.aggregates import stats
import numpy as np

from absl.testing import absltest


class StatsTest(absltest.TestCase):

  def test_stats_mean_variance_one_element(self):
    batches = np.array([np.random.randn(1) for _ in range(1)])
    state = stats.StatsState()
    for batch in batches:
      state.add(batch)
    np.testing.assert_allclose(state.mean, np.mean(batches))
    np.testing.assert_allclose(state.var, np.var(batches))
    np.testing.assert_allclose(state.stddev, np.std(batches))
    np.testing.assert_allclose(state.min, np.max(batches))
    np.testing.assert_allclose(state.max, np.min(batches))
    np.testing.assert_allclose(state.total, np.sum(batches))
    np.testing.assert_allclose(state.count, batches.size)

  def test_stats_mean_variance_with_two_elements(self):
    batches = np.array([np.random.randn(2) for _ in range(1)])
    state = stats.StatsState()
    for batch in batches:
      state.add(batch)
    np.testing.assert_allclose(state.mean, np.mean(batches))
    np.testing.assert_allclose(state.var, np.var(batches))
    np.testing.assert_allclose(state.stddev, np.std(batches))
    np.testing.assert_allclose(state.max, np.max(batches))
    np.testing.assert_allclose(state.min, np.min(batches))
    np.testing.assert_allclose(state.total, np.sum(batches))
    np.testing.assert_allclose(state.count, batches.size)

  def test_stats_mean_variance_with_batches(self):
    batches = np.array([np.random.randn(30) for _ in range(30)])
    state = stats.StatsState()
    for batch in batches:
      state.add(batch)
    np.testing.assert_allclose(state.mean, np.mean(batches))
    np.testing.assert_allclose(state.var, np.var(batches))
    np.testing.assert_allclose(state.stddev, np.std(batches))
    np.testing.assert_allclose(state.max, np.max(batches))
    np.testing.assert_allclose(state.min, np.min(batches))
    np.testing.assert_allclose(state.total, np.sum(batches))
    np.testing.assert_allclose(state.count, batches.size)

  def test_stats_mean_variance_with_function(self):
    batches = np.array([np.random.randn(30) for _ in range(30)])
    fn = lambda x: x + 1
    state = stats.StatsState(batch_score_fn=fn)
    for batch in batches:
      state.add(batch)
    np.testing.assert_allclose(state.mean, np.mean(fn(batches)))
    np.testing.assert_allclose(state.var, np.var(fn(batches)))
    np.testing.assert_allclose(state.stddev, np.std(fn(batches)))
    np.testing.assert_allclose(state.max, np.max(fn(batches)))
    np.testing.assert_allclose(state.min, np.min(fn(batches)))
    np.testing.assert_allclose(state.total, np.sum(fn(batches)))
    np.testing.assert_allclose(state.count, batches.size)


if __name__ == "__main__":
  absltest.main()
