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
"""Tests for rolling_stats.py."""

import dataclasses
import math
from typing import Any

from absl.testing import parameterized
from ml_metrics._src.aggregates import rolling_stats
import numpy as np

from absl.testing import absltest


def get_expected_stats_state(batches, batch_score_fn=None):
  if batch_score_fn is not None:
    batches = [batch_score_fn(batch) for batch in batches]
  batches = np.asarray(batches)
  batch = np.asarray(batches).reshape(-1, *batches.shape[2:])
  return rolling_stats.MeanAndVariance(
      batch_score_fn=batch_score_fn,
      _mean=np.nanmean(batch, axis=0),
      _var=np.nanvar(batch, axis=0),
      _count=np.nansum(~np.isnan(batch), axis=0),
  )


class StatsStateTest(parameterized.TestCase):

  def assertDataclassAlmostEqual(
      self, expected: dict[str, Any], got: dict[str, Any]
  ):
    """Helper function for using assertAlmostEquals in dictionaries."""
    expected = dataclasses.asdict(expected)
    got = dataclasses.asdict(got)
    self.assertEqual(expected.keys(), got.keys())
    for key, value in expected.items():
      if key == 'batch_score_fn':
        self.assertAlmostEqual(value, got[key])
      else:
        np.testing.assert_allclose(value, got[key])

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
    state = rolling_stats.MeanAndVariance()

    last_batch_result = None
    for batch in batches:
      last_batch_result = state.add(batch)
    self.assertIsInstance(last_batch_result, rolling_stats.MeanAndVariance)

    expected_last_batch_result = get_expected_stats_state(batches[-1])
    self.assertDataclassAlmostEqual(
        expected_last_batch_result, last_batch_result
    )

    expected_state = get_expected_stats_state(batches)
    self.assertDataclassAlmostEqual(expected_state, state)

  @parameterized.named_parameters([
      dict(
          testcase_name='all_nan',
          partial_nan=False,
      ),
      dict(
          testcase_name='partial_nan',
          partial_nan=True,
      ),
  ])
  def test_stats_state_batch_with_nan(self, partial_nan):
    batch = [np.nan] * 3 + [1, 2, 3] * int(partial_nan)
    state = rolling_stats.MeanAndVariance()
    state.add(batch)

    expected_state = get_expected_stats_state(batch)
    self.assertDataclassAlmostEqual(expected_state, state)

  def test_stats_state_invalid_batch_score_fn(self):
    batch = [1, 2, 3]
    state = rolling_stats.MeanAndVariance(batch_score_fn=lambda x: [0])
    with self.assertRaisesRegex(
        ValueError,
        'The `batch_score_fn` must return a series of the same length as the'
        ' `batch`.',
    ):
      state.add(batch)

  def test_stats_state_with_batch_score_fn(self):
    batches = np.array([np.random.randn(30) for _ in range(30)])
    batch_score_fn = lambda x: x + 1

    state = rolling_stats.MeanAndVariance(batch_score_fn=batch_score_fn)
    last_batch_result = None
    for batch in batches:
      last_batch_result = state.add(batch)

    expected_last_batch_state = get_expected_stats_state(
        batches[-1], batch_score_fn
    )
    self.assertDataclassAlmostEqual(
        expected_last_batch_state, last_batch_result
    )

    expected_state = get_expected_stats_state(batches, batch_score_fn)
    self.assertDataclassAlmostEqual(expected_state, state)

  @parameterized.named_parameters([
      dict(
          testcase_name='merge_nonempty_states',
          is_self_empty=False,
          is_other_empty=False,
      ),
      dict(
          testcase_name='merge_nonempty_with_empty_state',
          is_self_empty=False,
          is_other_empty=True,
      ),
      dict(
          testcase_name='merge_empty_with_nonempty_state',
          is_self_empty=True,
          is_other_empty=False,
      ),
      dict(
          testcase_name='merge_empty_states',
          is_self_empty=True,
          is_other_empty=True,
      ),
  ])
  def test_stats_state_merge(self, is_self_empty, is_other_empty):
    self_batch = np.random.randn(30 * int(not is_self_empty))
    other_batch = np.random.randn(30 * int(not is_other_empty))
    self_state = rolling_stats.MeanAndVariance()
    other_state = rolling_stats.MeanAndVariance()

    if not is_self_empty:
      self_state.add(self_batch)
    if not is_other_empty:
      other_state.add(other_batch)
    self_state.merge(other_state)

    expected_state = get_expected_stats_state(
        np.append(self_batch, other_batch)
    )
    self.assertDataclassAlmostEqual(expected_state, self_state)

  @parameterized.named_parameters([
      dict(
          testcase_name='init_state',
          add_batch=False,
      ),
      dict(
          testcase_name='non_empty_batch',
          add_batch=True,
      ),
  ])
  def test_stats_state_properties(self, add_batch):
    batch = [1, 2, 3, np.nan] * int(add_batch)
    state = rolling_stats.MeanAndVariance()
    if add_batch:
      state.add(batch)

    expected_properties_dict = {
        'mean': np.nanmean(batch, axis=0),
        'var': np.nanvar(batch, axis=0),
        'stddev': np.nanstd(batch, axis=0),
        'count': np.nansum(~np.isnan(batch), axis=0),
        'total': np.nansum(batch, axis=0),
    }
    for property_name, value in expected_properties_dict.items():
      np.testing.assert_allclose(getattr(state.result(), property_name), value)


class R2TjurTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='absolute', r2_metric=rolling_stats.R2Tjur),
      dict(testcase_name='relative', r2_metric=rolling_stats.R2TjurRelative),
  )
  def test_r2_tjur_merge(self, r2_metric):
    y_true_1 = (0, 1)
    y_pred_1 = (0.8, 0.3)

    y_true_2 = 1
    y_pred_2 = 0.9

    state_1 = r2_metric().add(y_true_1, y_pred_1)
    state_2 = r2_metric().add(y_true_2, y_pred_2)
    result = state_1.merge(state_2)

    # sum(y_true) = 0.0 + 1.0 + 1.0  = 2.0
    # sum(y_true * y_pred) = 0.0 * 0.8 + 1.0 * 0.3 + 1.0 * 0.9 = 1.2
    # sum(1 - y_true) = 1.0 + 0.0 + 0.0 = 1.0
    # sum((1 - y_true) * y_pred) = 1.0 * 0.8 + 0.0 * 0.3 + 0.0 * 0.9 = 0.8

    expected_result = r2_metric(
        sum_y_true=2.0, sum_y_pred=1.2, sum_neg_y_true=1.0, sum_neg_y_pred=0.8
    )

    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      # sum_y_true = 2.0, sum_y_pred = 1.2
      # sum_neg_y_true = 1.0, sum_neg_y_pred = 0.1
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=(1.2 / 2.0) - (0.8 / 1.0),
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=(1.2 / 2.0) / (0.8 / 1.0),
      ),
  )
  def test_r2_tjur_simple(self, r2_metric, expected_result):
    y_true = (0, 1, 1)
    y_pred = (0.8, 0.3, 0.9)

    actual_result = r2_metric().add(y_true, y_pred).result()

    self.assertAlmostEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=-9.78031226162579e-05,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=0.9998043715885978,
      ),
  )
  def test_r2_tjur_one_large_batch(self, r2_metric, expected_result):
    np.random.seed(seed=0)

    y_true = np.random.rand(1000000)
    y_pred = np.random.rand(1000000)

    actual_result = r2_metric().add(y_true, y_pred).result()

    self.assertAlmostEqual(actual_result, expected_result, places=14)

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=-2340.933593683032,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=1.0129655857337465,
      ),
  )
  def test_r2_tjur_many_batches_little_correlation(
      self, r2_metric, expected_result
  ):
    np.random.seed(seed=0)

    y_true = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))
    y_pred = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = r2_metric()
    for y_true_i, y_pred_i in zip(y_true, y_pred):
      state.add(y_true_i, y_pred_i)

    self.assertAlmostEqual(state.result(), expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=0.9999901799905766,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=214595.2102587055,
      ),
  )
  def test_r2_tjur_many_batches_much_correlation(
      self, r2_metric, expected_result
  ):
    np.random.seed(seed=0)

    y_true = np.random.uniform(low=1e-5, high=1 - 1e-5, size=(10000, 10000))

    # This is a noisy version of y_true.
    y_pred = y_true + np.random.uniform(
        low=-1e-5, high=1e-5, size=(10000, 10000)
    )

    y_true = np.round(y_true)
    y_pred = np.round(y_pred)

    state = r2_metric()
    for y_true_i, y_pred_i in zip(y_true, y_pred):
      state.add(y_true_i, y_pred_i)

    self.assertAlmostEqual(state.result(), expected_result, places=10)

  def test_r2_tjur_absolute_many_batches_direct_correlation(self):
    y = np.round(np.random.uniform(size=(10000, 10000)))

    state = rolling_stats.R2Tjur()
    for y_i in y:
      state.add(y_i, y_i)

    expected_result = 1

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  def test_r2_tjur_relative_many_batches_direct_correlation(self):
    y = np.round(np.random.uniform(size=(10000, 10000)))

    state = rolling_stats.R2TjurRelative()
    for y_i in y:
      state.add(y_i, y_i)

    self.assertTrue(math.isnan(state.result()))

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=-1,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=0,
      ),
  )
  def test_r2_tjur_many_batches_inverse_correlation(
      self, r2_metric, expected_result
  ):
    y = np.round(np.random.uniform(size=(10000, 10000)))

    state = r2_metric()
    for y_i in y:
      state.add(y_i, 1 - y_i)

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  @parameterized.named_parameters(
      dict(testcase_name='y_true_is_0', y_true=(0, 0, 0)),
      dict(testcase_name='y_true_is_1', y_true=(1, 1, 1)),
  )
  def test_r2_tjur_absolute_returns_nan(self, y_true):
    y_pred = (1, 0, 1)

    self.assertTrue(
        math.isnan(rolling_stats.R2Tjur().add(y_true, y_pred).result())
    )

  @parameterized.named_parameters(
      dict(testcase_name='y_true_is_0', y_true=(0, 0, 0), y_pred=(1, 0, 1)),
      dict(
          testcase_name='sum_neg_y_pred_is_0',
          # all(y_t == 1 or y_p == 0 for y_t, y_p in zip(y_true, y_pred))
          y_true=(1, 0, 1),
          y_pred=(1, 0, 1),
      ),
  )
  def test_r2_tjur_relative_returns_nan(self, y_true, y_pred):
    self.assertTrue(
        math.isnan(rolling_stats.R2TjurRelative().add(y_true, y_pred).result())
    )


class RRegressionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='centered', center=True),
      dict(testcase_name='not_centered', center=False),
  )
  def test_r_regression_merge(self, center):
    x_1 = (1, 2, 3, 4)
    y_1 = (10, 9, 2.5, 6)

    x_2 = (5, 6, 7)
    y_2 = (4, 3, 2)

    state_1 = rolling_stats.RRegression(center=center).add(x_1, y_1)
    state_2 = rolling_stats.RRegression(center=center).add(x_2, y_2)
    result = state_1.merge(state_2)

    expected_result = rolling_stats.RRegression(
        num_samples=7,  # len(x_1) + len(x_2)
        sum_x=28,  # 1 + 2 + 3 + 4 + 5 + 6 + 7
        sum_y=36.5,  # 10 + 9 + 2.5 + 6 + 4 + 3 + 2
        sum_xx=140,  # 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2
        sum_yy=252.25,  # 10^2 + 9^2 + 2.5^2 + 6^2 + 4^2 + 3^2 + 2^2
        sum_xy=111.5,  # 1*10 + 2*9 + 3*2.5 + 4*6 + 5*4 + 6*3 + 7*2
    )

    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='centered',
          center=True,
          # From
          # sklearn.feature_selection.r_regression(
          # X=np.reshape(x, (-1, 1)), y=y
          # )[0]
          expected_result=-0.8285038835884279,
      ),
      dict(
          testcase_name='not_centered',
          center=False,
          # From
          # sklearn.feature_selection.r_regression(
          # X=np.reshape(x, (-1, 1)), y=y, center=False
          # )[0]
          expected_result=0.5933285714624903,
      ),
  )
  def test_r_regression_single_output(self, center, expected_result):
    x = (1, 2, 3, 4, 5, 6, 7)
    y = (10, 9, 2.5, 6, 4, 3, 2)

    actual_result = rolling_stats.RRegression(center=center).add(x, y).result()

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  @parameterized.named_parameters(
      dict(
          testcase_name='centered',
          center=True,
          # From sklearn.feature_selection.r_regression(x_all, y)
          expected_result=(-0.82850388, -0.32338709),
      ),
      dict(
          testcase_name='not_centered',
          center=False,
          # From
          # sklearn.feature_selection.r_regression(X=x_all, y=y, center=False)
          expected_result=(0.59332857, 0.72301752),
      ),
  )
  def test_r_regression_multi_output(self, center, expected_result):
    x1 = (10, 9, 2.5, 6, 4, 3, 2)
    x2 = (8, 6, 7, 5, 3, 0, 9)
    y = (1, 2, 3, 4, 5, 6, 7)

    x_all = np.array((x1, x2)).T

    actual_result = (
        rolling_stats.RRegression(center=center).add(x_all, y).result()
    )

    np.testing.assert_almost_equal(actual_result, expected_result)

  def test_r_regression_one_large_batch(self):
    np.random.seed(seed=0)

    x = np.random.rand(1000000)
    y = np.random.rand(1000000)

    actual_result = rolling_stats.RRegression().add(x, y).result()

    # From
    # sklearn.feature_selection.r_regression(X=np.reshape(x, (-1, 1)), y=y)[0]
    expected_result = -0.0002932187695762664

    self.assertAlmostEqual(actual_result, expected_result, places=13)

  def test_r_regression_many_batches_little_correlation(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))
    y = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    # From
    # sklearn.feature_selection.r_regression(X=np.reshape(x, (-1, 1)), y=y)[0]
    expected_result = 4.231252166807617e-05

    self.assertAlmostEqual(state.result(), expected_result, places=14)

  def test_r_regression_many_batches_much_correlation(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    # This is a noisy version of x.
    y = x + np.random.uniform(low=-1e5, high=1e5, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    # From
    # sklearn.feature_selection.r_regression(X=np.reshape(x, (-1, 1)), y=y)[0]
    expected_result = 0.995037725730923

    self.assertAlmostEqual(state.result(), expected_result, places=10)

  def test_r_regression_many_batches_direct_correlation(self):
    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i in x:
      state.add(x_i, x_i)

    expected_result = 1

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  def test_r_regression_many_batches_inverse_correlation(self):
    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i in x:
      state.add(x_i, -x_i)

    expected_result = -1

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  @parameterized.named_parameters(
      dict(testcase_name='empty_input', x=(), y=()),
      dict(testcase_name='0_input', x=(0, 0), y=(0, 0)),
  )
  def test_r_regression_returns_nan(self, x, y):
    self.assertTrue(math.isnan(rolling_stats.RRegression().add(x, y).result()))

  def test_r_regression_valid_and_0_input(self):
    x_valid = (10, 9, 2.5, 6, 4, 3, 2)
    x_0 = (0, 0, 0, 0, 0, 0, 0)
    y = (1, 2, 3, 4, 5, 6, 7)

    x_all = np.array((x_valid, x_0)).T

    actual_result = rolling_stats.RRegression().add(x_all, y).result()

    # From sklearn.feature_selection.r_regression(x_all, y)
    expected_result = (-0.82850388, float('nan'))

    np.testing.assert_almost_equal(actual_result, expected_result)


class SymmetricPredictionDifferenceTest(parameterized.TestCase):

  def test_symmetric_prediction_difference_merge(self):
    x_1 = (0, 1)
    y_1 = (0.8, 0.3)

    x_2 = 1
    y_2 = 0.9

    state_1 = rolling_stats.SymmetricPredictionDifference().add(x_1, y_1)
    state_2 = rolling_stats.SymmetricPredictionDifference().add(x_2, y_2)
    result = state_1.merge(state_2)

    # sum_half_pointwise_rel_diff =
    # (|0 - 0.8| / |0 + 0.8|
    # + |1 - 0.3| / |1 + 0.3|
    # + |1 - 0.9| / |1 + 0.9|)
    # = 1.59109311741

    self.assertEqual(result.num_samples, 3)
    self.assertAlmostEqual(
        result.sum_half_pointwise_rel_diff, 1.59109311741, places=11
    )

  def test_symmetric_prediction_difference_simple(self):
    x = (0, 1, 1)
    y = (0.8, 0.3, 0.9)

    expected_result = 1.06072874494  # 2 * 1.59109311741 / 3 = 1.06072874494

    actual_result = (
        rolling_stats.SymmetricPredictionDifference().add(x, y).result()
    )

    self.assertAlmostEqual(actual_result, expected_result, places=11)

  def test_symmetric_prediction_difference_one_large_batch(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=1000000)
    y = np.random.uniform(low=-1e6, high=1e6, size=1000000)

    actual_result = (
        rolling_stats.SymmetricPredictionDifference().add(x, y).result()
    )

    expected_result = 32.611545081600894

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  def test_symmetric_prediction_difference_many_batches_little_correlation(
      self,
  ):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))
    y = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    expected_result = 36.23844148369455

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    self.assertAlmostEqual(state.result(), expected_result, places=10)

  def test_symmetric_prediction_difference_many_batches_much_correlation(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=1e-5, high=1 - 1e-5, size=(10000, 10000))

    # y is a noisy version of x.
    y = x + np.random.uniform(low=-1e-5, high=1e-5, size=(10000, 10000))

    expected_result = 5.8079104443249064e-05

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    self.assertAlmostEqual(state.result(), expected_result, places=16)

  def test_symmetric_prediction_difference_many_identical_batches(self):
    x = np.random.uniform(size=(10000, 10000))

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i in x:
      # sum_half_pointwise_rel_diff should be 0 for every point.
      state.add(x_i, x_i)

    expected_result = 0

    self.assertAlmostEqual(state.result(), expected_result, places=11)

  def test_symmetric_prediction_difference_many_batches_opposite(self):
    x = np.random.uniform(size=(10000, 10000))

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i in x:
      # sum_half_pointwise_rel_diff should remain 0 because all of the pointwise
      # average relative differences are undefined.
      state.add(x_i, -x_i)

    expected_result = 0

    self.assertAlmostEqual(state.result(), expected_result, places=11)

  def test_symmetric_prediction_difference_absolute_returns_nan(self):
    x_empty = ()

    self.assertTrue(
        math.isnan(
            rolling_stats.SymmetricPredictionDifference()
            .add(x_empty, x_empty)
            .result()
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='k_epsilon_1e-7',
          k_epsilon=1e-7,
          # 2e-8 + 1e-8 = 3e-8 < k_epsilon = 1e-7.
          # 2 * (1 - 0.3) / (1 + 0.3) / 3 = 0.358974358974
          expected_result=0.358974358974,
      ),
      dict(
          testcase_name='k_epsilon_3e-8',
          k_epsilon=3e-8,
          # 1e-8 + 2e-8 = 3e-8 = k_epsilon = 1e-8.
          # 2 * ((1 - 0.3) / (1 + 0.3) + (2e-8 - 1e-8) / (2e-8 + 1e-8)) / 3
          # 2 * (0.7 / 1.3 + 1 / 3) / 3 = 0.581196581197
          expected_result=0.581196581197,
      ),
      dict(
          testcase_name='k_epsilon_1e-9',
          k_epsilon=1e-9,
          # 2e-8 + 1e-8 = 3e-8 > k_epsilon = 1e-9.
          expected_result=0.581196581197,
      ),
  )
  def test_symmetric_prediction_difference_k_epsilon_single_small_batch(
      self, k_epsilon, expected_result
  ):
    x = (0, 1, 1e-8)
    y = (0, 0.3, 2e-8)

    actual_result = (
        rolling_stats.SymmetricPredictionDifference(k_epsilon=k_epsilon)
        .add(x, y)
        .result()
    )

    self.assertAlmostEqual(actual_result, expected_result, places=12)

  @parameterized.named_parameters(
      dict(testcase_name='within_k_epsilon', data_max=1e-8, expected_result=0),
      dict(
          testcase_name='partially_outside_k_epsilon',
          data_max=1e-6,
          expected_result=3.190964660292838,
      ),
  )
  def test_symmetric_prediction_difference_k_epsilon_many_large_batches(
      self, data_max, expected_result
  ):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-data_max, high=data_max, size=(1000, 1000))
    y = np.random.uniform(low=-data_max, high=data_max, size=(1000, 1000))

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    self.assertAlmostEqual(state.result(), expected_result, places=11)

  def test_symmetric_prediction_difference_asserts_with_invalid_input(self):
    # x.shape != y.shape
    x = (1, 2, 3)
    y = (4, 5)

    expected_error_message = (
        r'SymmetricPredictionDifference\.add\(\) requires x and y to have the'
        r' same shape, but recieved x=\[1\. 2\. 3\.\] and y=\[4\. 5\.\] with'
        r' x.shape=\(3\,\) and y.shape=\(2\,\)'
    )

    metric = rolling_stats.SymmetricPredictionDifference()

    with self.assertRaisesRegex(ValueError, expected_error_message):
      metric.add(x, y)


if __name__ == '__main__':
  absltest.main()
