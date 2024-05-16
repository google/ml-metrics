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
import dataclasses
import math
from typing import Any

from absl.testing import parameterized
from ml_metrics._src.aggregates import stats
import numpy as np

from absl.testing import absltest


def get_expected_result(batches, batch_score_fn=None):
  if batch_score_fn is not None:
    batches = [batch_score_fn(batch) for batch in batches]
  batches = np.asarray(batches)
  batch = np.asarray(batches).reshape(-1, *batches.shape[2:])
  return stats.StatsState(
      batch_score_fn=batch_score_fn,
      _min=np.nanmin(batch, axis=0),
      _max=np.nanmax(batch, axis=0),
      _mean=np.nanmean(batch, axis=0),
      _var=np.nanvar(batch, axis=0),
      _count=np.nansum(~np.isnan(batch), axis=0),
  )


class StatsTest(parameterized.TestCase):

  def assertDataclassAlmostEqual(
      self, d1: dict[str, Any], d2: dict[str, Any], msg=None, places=7
  ):
    """Helper function for using assertAlmostEquals in dictionaries."""
    d1 = dataclasses.asdict(d1)
    d2 = dataclasses.asdict(d2)
    self.assertEqual(d1.keys(), d2.keys())
    for k, v in d1.items():
      self.assertEqual(type(v), type(d2[k]))
      if isinstance(v, dict):
        self.assertDataclassAlmostEqual(v, d2[k], msg=msg)
      else:
        self.assertAlmostEqual(v, d2[k], places=places, msg=msg)

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
    state = stats.StatsState()

    last_batch_result = None
    for batch in batches:
      last_batch_result = state.add(batch)
    self.assertIsInstance(last_batch_result, stats.StatsState)

    expected_last_batch_result = get_expected_result(batches[-1])
    self.assertDataclassAlmostEqual(
        expected_last_batch_result, last_batch_result
    )

    expected_result = get_expected_result(batches)
    self.assertDataclassAlmostEqual(expected_result, state.result())

  # TODO: b/311207032 - Add tests for nan inputs.
  # TODO: b/311207032 - Add tests for multi dimensional inputs.

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

    self.assertDataclassAlmostEqual(
        last_batch_result, expected_last_batch_result
    )
    self.assertDataclassAlmostEqual(state.result(), expected_result)

  def test_stats_state_merge(self):
    batches = np.array([np.random.randn(30) for _ in range(2)])
    expected_result = get_expected_result(batches)

    state_0 = stats.StatsState()
    state_0.add(batches[0])
    state_1 = stats.StatsState()
    state_1.add(batches[1])

    state_0.merge(state_1)
    result = state_0.result()

    self.assertDataclassAlmostEqual(expected_result, result)

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

    if self_empty:
      state_self.merge(state_other)
    else:
      state_self.merge(state_other)
    expected = stats.StatsState().add([1, 2, 3])
    self.assertEqual(expected, state_self)


class CoeffStateTest(parameterized.TestCase):

  def test_coeff_state_merge(self):
    x_1 = (1, 2, 3, 4)
    y_1 = (10, 9, 2.5, 6)

    x_2 = (5, 6, 7)
    y_2 = (4, 3, 2)

    new_state = stats._CoeffState()
    state_1 = new_state.from_inputs(x_1, y_1)
    state_2 = new_state.from_inputs(x_2, y_2)
    result = state_1.merge(state_2)

    expected_result = stats._CoeffState(
        num_samples=7,
        sum_x=28,
        sum_y=36.5,
        sum_xx=140,
        sum_yy=252.25,
        sum_xy=111.5,
    )

    self.assertEqual(result, expected_result)

  def test_coeff_agg_fn_base_merge_states(self):
    x_1 = (1, 2, 3, 4)
    y_1 = (10, 9, 2.5, 6)

    x_2 = (5, 6, 7)
    y_2 = (4, 3, 2)

    new_state = stats._CoeffState()
    state_1 = new_state.from_inputs(x_1, y_1)
    state_2 = new_state.from_inputs(x_2, y_2)

    new_agg_fn = stats.PearsonCorrelationCoefficientAggFn()
    result = new_agg_fn.merge_states((state_1, state_2))

    expected_result = stats._CoeffState(
        num_samples=7,
        sum_x=28,
        sum_y=36.5,
        sum_xx=140,
        sum_yy=252.25,
        sum_xy=111.5,
    )

    self.assertEqual(result, expected_result)

  def test_pearson_correlation_coefficient_simple(self):
    x = (1, 2, 3, 4, 5, 6, 7)
    y = (10, 9, 2.5, 6, 4, 3, 2)

    actual_result = stats.PearsonCorrelationCoefficientAggFn()(x, y)

    # From scipy.stats.pearsonr(x=x, y=y).statistic
    expected_result = -0.8285038835884279

    self.assertAlmostEqual(actual_result, expected_result)

  def test_pearson_correlation_coefficient_one_batch(self):
    np.random.seed(seed=0)
    x = np.random.rand(1000000)
    y = np.random.rand(1000000)

    actual_result = stats.PearsonCorrelationCoefficientAggFn()(x, y)

    # From scipy.stats.pearsonr(x=x, y=y).statistic
    expected_result = -0.00029321876957677745

    self.assertAlmostEqual(actual_result, expected_result)

  def test_pearson_correlation_coefficient_many_batches_little_correlation(
      self,
  ):
    np.random.seed(seed=0)
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])
    y = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])

    state = stats.PearsonCorrelationCoefficientAggFn().create_state()
    for x_i, y_i in zip(x, y):
      stats.PearsonCorrelationCoefficientAggFn().update_state(state, x_i, y_i)

    actual_result = stats.PearsonCorrelationCoefficientAggFn().get_result(state)

    # From scipy.stats.pearsonr(x=x, y=y).statistic
    expected_result = 4.231252166809374e-05

    self.assertAlmostEqual(actual_result, expected_result, places=15)

  def test_pearson_correlation_coefficient_many_batches_much_correlation(self):
    np.random.seed(seed=0)
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])
    y = x + np.array([
        np.random.uniform(low=-1e5, high=1e5, size=10000) for _ in range(10000)
    ])  # This is a noisy version of x.

    state = stats.PearsonCorrelationCoefficientAggFn().create_state()
    for x_i, y_i in zip(x, y):
      stats.PearsonCorrelationCoefficientAggFn().update_state(state, x_i, y_i)

    actual_result = stats.PearsonCorrelationCoefficientAggFn().get_result(state)

    # From scipy.stats.pearsonr(x=x, y=y).statistic
    expected_result = 0.9950377257308471

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  def test_pearson_correlation_coefficient_many_batches_direct_correlation(
      self,
  ):
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])

    state = stats.PearsonCorrelationCoefficientAggFn().create_state()
    for x_i in x:
      stats.PearsonCorrelationCoefficientAggFn().update_state(state, x_i, x_i)

    actual_result = stats.PearsonCorrelationCoefficientAggFn().get_result(state)

    expected_result = 1

    self.assertAlmostEqual(actual_result, expected_result, places=15)

  def test_pearson_correlation_coefficient_many_batches_inverse_correlation(
      self,
  ):
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])

    state = stats.PearsonCorrelationCoefficientAggFn().create_state()
    for x_i in x:
      stats.PearsonCorrelationCoefficientAggFn().update_state(state, x_i, -x_i)

    actual_result = stats.PearsonCorrelationCoefficientAggFn().get_result(state)

    expected_result = -1

    self.assertAlmostEqual(actual_result, expected_result, places=15)

  @parameterized.named_parameters(
      dict(testcase_name='empty_input', x=(), y=()),
      dict(testcase_name='0_input', x=(0, 0), y=(0, 0)),
  )
  def test_pearson_correlation_coefficient_returns_nan(self, x, y):
    self.assertTrue(
        math.isnan(stats.PearsonCorrelationCoefficientAggFn()(x, y))
    )

  def test_coefficient_of_determination_simple(self):
    x = (1, 2, 3, 4, 5, 6, 7)
    y = (10, 9, 2.5, 6, 4, 3, 2)

    actual_result = stats.CoefficientOfDeterminationAggFn()(x, y)

    # From scipy.stats.pearsonr(x=x, y=y).statistic ** 2
    expected_result = 0.6864186851211073

    self.assertAlmostEqual(actual_result, expected_result)

  def test_coefficient_of_determination_one_batch(self):
    np.random.seed(seed=0)
    x = np.random.uniform(low=-1e6, high=1e6, size=1000000)
    y = np.random.uniform(low=-1e6, high=1e6, size=1000000)

    actual_result = stats.CoefficientOfDeterminationAggFn()(x, y)

    # From scipy.stats.pearsonr(x=x, y=y).statistic ** 2
    expected_result = 8.597724683211943e-08

    self.assertAlmostEqual(actual_result, expected_result)

  def test_coefficient_of_determination_many_batches_little_correlation(
      self,
  ):
    np.random.seed(seed=0)
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])
    y = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])

    state = stats.CoefficientOfDeterminationAggFn().create_state()
    for x_i, y_i in zip(x, y):
      stats.CoefficientOfDeterminationAggFn().update_state(state, x_i, y_i)

    actual_result = stats.CoefficientOfDeterminationAggFn().get_result(state)

    # From scipy.stats.pearsonr(x=x, y=y).statistic ** 2
    expected_result = 1.7903494899129026e-09

    self.assertAlmostEqual(actual_result, expected_result, places=15)

  def test_coefficient_of_determination_many_batches_much_correlation(self):
    np.random.seed(seed=0)
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])
    y = x + np.array([
        np.random.uniform(low=-1e5, high=1e5, size=10000) for _ in range(10000)
    ])  # This is a noisy version of x.

    state = stats.CoefficientOfDeterminationAggFn().create_state()
    for x_i, y_i in zip(x, y):
      stats.CoefficientOfDeterminationAggFn().update_state(state, x_i, y_i)

    actual_result = stats.CoefficientOfDeterminationAggFn().get_result(state)

    # From scipy.stats.pearsonr(x=x, y=y).statistic ** 2
    expected_result = 0.9901000756276166

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  def test_coefficient_of_determination_many_batches_direct_correlation(
      self,
  ):
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])

    state = stats.CoefficientOfDeterminationAggFn().create_state()
    for x_i in x:
      stats.CoefficientOfDeterminationAggFn().update_state(state, x_i, x_i)

    actual_result = stats.CoefficientOfDeterminationAggFn().get_result(state)

    expected_result = 1

    self.assertAlmostEqual(actual_result, expected_result, places=15)

  def test_coefficient_of_determination_many_batches_inverse_correlation(
      self,
  ):
    x = np.array([
        np.random.uniform(low=-1e6, high=1e6, size=10000) for _ in range(10000)
    ])

    state = stats.CoefficientOfDeterminationAggFn().create_state()
    for x_i in x:
      stats.CoefficientOfDeterminationAggFn().update_state(state, x_i, -x_i)

    actual_result = stats.CoefficientOfDeterminationAggFn().get_result(state)

    expected_result = 1

    self.assertAlmostEqual(actual_result, expected_result, places=15)

  @parameterized.named_parameters(
      dict(testcase_name='empty_input', x=(), y=()),
      dict(testcase_name='0_input', x=(0, 0), y=(0, 0)),
  )
  def test_coefficient_of_determination_returns_nan(self, x, y):
    self.assertTrue(math.isnan(stats.CoefficientOfDeterminationAggFn()(x, y)))


if __name__ == '__main__':
  absltest.main()
