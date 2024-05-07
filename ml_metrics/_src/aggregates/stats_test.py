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

import math

from absl.testing import parameterized
from ml_metrics._src.aggregates import stats
import numpy as np

from absl.testing import absltest


def get_expected_result(batch, batch_score_fn=None):
  if batch_score_fn is not None:
    batch = batch_score_fn(batch)
  # result_dict key must match StatsState properties name.
  result_dict = {
      'min': np.min(batch),
      'max': np.max(batch),
      'mean': np.mean(batch),
      'var': np.var(batch),
      'stddev': np.std(batch),
      'count': batch.size,
      'total': np.sum(batch),
  }
  return result_dict


class StatsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='1_batch_1_element',
          num_batch=1,
          num_elements_per_batch=1,
      ),
      dict(
          testcase_name='1_batch_2_element',
          num_batch=1,
          num_elements_per_batch=2,
      ),
      dict(
          testcase_name='1_batch_1000_element',
          num_batch=1,
          num_elements_per_batch=1000,
      ),
      dict(
          testcase_name='2_batch_1_element',
          num_batch=2,
          num_elements_per_batch=1,
      ),
      dict(
          testcase_name='2_batch_2_element',
          num_batch=2,
          num_elements_per_batch=2,
      ),
      dict(
          testcase_name='2_batch_1000_element',
          num_batch=2,
          num_elements_per_batch=1000,
      ),
      dict(
          testcase_name='1000_batch_1_element',
          num_batch=1000,
          num_elements_per_batch=1,
      ),
      dict(
          testcase_name='1000_batch_2_element',
          num_batch=1000,
          num_elements_per_batch=2,
      ),
      dict(
          testcase_name='1000_batch_1000_element',
          num_batch=1000,
          num_elements_per_batch=1000,
      ),
  ])
  def test_stats_state(self, num_batch, num_elements_per_batch):
    batches = np.array(
        [np.random.randn(num_elements_per_batch) for _ in range(num_batch)]
    ) + 1e7
    expected_result = get_expected_result(batches)
    expected_last_batch_result = get_expected_result(batches[-1])
    state = stats.StatsState()

    last_batch_result = None
    for batch in batches:
      last_batch_result = state.add(batch)
    self.assertIsInstance(last_batch_result, stats.StatsState)

    for metric in expected_last_batch_result:
      self.assertAlmostEqual(
          expected_last_batch_result[metric], getattr(last_batch_result, metric)
      )

    result = state.result()

    self.assertIsInstance(result, stats.StatsState)
    for metric in expected_result:
      np.testing.assert_allclose(
          expected_result[metric], getattr(result, metric)
      )

  def test_stats_state_initial_add_with_empty_batch(self):
    state = stats.StatsState()
    with self.assertRaisesRegex(
        ValueError, '`batch` must not be empty.'
    ):
      state.add([])

  def test_stats_state_with_score_fn(self):
    batches = np.array([np.random.randn(30) for _ in range(30)])
    batch_score_fn = lambda x: x + 1
    expected_result = get_expected_result(batches, batch_score_fn)
    expected_last_batch_result = get_expected_result(
        batches[-1], batch_score_fn
    )
    state = stats.StatsState(batch_score_fn=batch_score_fn)

    last_batch_result = None
    for batch in batches:
      last_batch_result = state.add(batch)
    self.assertIsInstance(last_batch_result, stats.StatsState)

    for metric in expected_last_batch_result:
      self.assertAlmostEqual(
          expected_last_batch_result[metric], getattr(last_batch_result, metric)
      )

    result = state.result()

    self.assertIsInstance(result, stats.StatsState)
    for metric in expected_result:
      np.testing.assert_allclose(
          expected_result[metric], getattr(result, metric)
      )

  def test_stats_state_merge(self):
    batches = np.array([np.random.randn(30) for _ in range(2)])
    expected_result = get_expected_result(batches)

    state_0 = stats.StatsState()
    state_0.add(batches[0])
    state_1 = stats.StatsState()
    state_1.add(batches[1])

    state_0.merge(state_1)
    result = state_0.result()

    self.assertIsInstance(result, stats.StatsState)
    for metric in expected_result:
      np.testing.assert_allclose(
          expected_result[metric], getattr(result, metric)
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='merge_empty_state',
          self_empty=False,
      ),
      dict(
          testcase_name='merge_to_empty_state',
          self_empty=True,
      ),
  ])
  def test_stats_state_merge_empty_state_stats(self, self_empty):
    state_self = stats.StatsState()
    state_other = stats.StatsState()
    if self_empty:
      state_other.add([1, 2, 3])
    else:
      state_self.add([1, 2, 3])

    with self.assertRaisesRegex(
        ValueError, 'Both StatsStates must not be empty.'
    ):
      state_self.merge(state_other)

  def test_pearson_correlation_coefficient_one_batch(self):
    x = ((1, 2, 3, 4, 5, 6, 7),)
    y = ((10, 9, 2.5, 6, 4, 3, 2),)

    actual_result = stats.PearsonCorrelationCoefficientAggFn()(x, y)
    expected_result = -0.8285038835884279

    self.assertAlmostEqual(actual_result, expected_result)

  def test_pearson_correlation_coefficient_many_batches(self):
    np.random.seed(seed=0)
    x = np.random.rand(1000000)
    y = np.random.rand(1000000)

    expected_result = -0.00029321876957677745
    actual_result = stats.PearsonCorrelationCoefficientAggFn()(x, y)

    self.assertAlmostEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      dict(testcase_name='empty_input', x=(), y=()),
      dict(testcase_name='0_input', x=((0, 0),), y=((0, 0),)),
  )
  def test_pearson_correlation_coefficient_returns_nan(self, x, y):
    self.assertTrue(
        math.isnan(stats.PearsonCorrelationCoefficientAggFn()(x, y))
    )

  def test_pearson_correlation_coefficient_merge(self):
    x_1 = (1, 2, 3, 4)
    y_1 = (10, 9, 2.5, 6)

    x_2 = (5, 6, 7)
    y_2 = (4, 3, 2)

    expected_result = stats._PccState(
        num_samples=7,
        sum_x=28,
        sum_y=36.5,
        sum_xx=140,
        sum_yy=252.25,
        sum_xy=111.5,
    )

    new_state = stats._PccState()
    state_1 = new_state.from_inputs(x_1, y_1)
    state_2 = new_state.from_inputs(x_2, y_2)

    result = state_1.merge(state_2)

    self.assertEqual(result, expected_result)

if __name__ == '__main__':
  absltest.main()
