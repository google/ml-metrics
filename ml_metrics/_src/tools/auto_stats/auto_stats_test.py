import collections

from ml_metrics._src.tools.auto_stats import auto_stats
import numpy as np

from absl.testing import absltest


class AutoStatsTest(absltest.TestCase):

  def test_get_float_data_stats_small_input(self):
    input_data = (8, 6, 7, 5, 3, 0, 9)

    expected_stats = {
        'Count': 7,
        'Counter': {8: 1, 6: 1, 7: 1, 5: 1, 3: 1, 0: 1, 9: 1},
        'Histogram': (
            # Histogram values.
            np.array((1, 0, 0, 1, 0, 1, 1, 1, 1, 1)),
            # Bin edges.
            np.array((0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9)),
        ),
        'Max': 9,
        'Mean': 38 / 7,  # sum(input_data) / len(input_data)
        'Min': 0,
        'Standard Deviation': 2.8713930346059686,  # np.nanstd(input_data)
        'Variance': 8.244897959183673,  # np.nanvar(input_data)
    }

    actual_stats = auto_stats.get_float_data_stats(input_data)

    np.testing.assert_equal(actual_stats, expected_stats)

  def test_get_float_data_stats_large_input(self):
    np.random.seed(123)

    count = 10000
    input_data = np.random.random_sample(count) * 1000 - 500

    expected_stats = {
        'Count': count,
        'Counter': collections.Counter(input_data),
        'Histogram': np.histogram(input_data),
        'Max': 499.8900645459988,  # max(input_data)
        'Mean': -1.7688546896545108,  # np.mean(input_data)
        'Min': -499.93216168772494,  # min(input_data)
        'Standard Deviation': 288.2768533832462,  # np.nanstd(input_data)
        'Variance': 83103.54419654563,  # np.nanvar(input_data)
    }

    actual_stats = auto_stats.get_float_data_stats(input_data)

    np.testing.assert_equal(actual_stats, expected_stats)


if __name__ == '__main__':
  absltest.main()
