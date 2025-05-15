"""Auto Stats for MLAT.

This file calculates relevant statistics for a given input data shape.
"""

from collections.abc import Sequence
from typing import Any

from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.aggregates import types


def get_float_data_stats(data: Sequence[types.NumbersT]) -> dict[str, Any]:
  """Returns relevant statistics for float data."""
  mean_and_var = rolling_stats.MeanAndVariance().as_agg_fn()(data)
  min_max_and_count = rolling_stats.MinMaxAndCount().as_agg_fn()(data)
  return {
      'Count': min_max_and_count.count,
      'Counter': rolling_stats.Counter().as_agg_fn()(data),
      'Histogram': (
          rolling_stats.Histogram(
              range=(min_max_and_count.min, min_max_and_count.max)
          ).as_agg_fn()(data)
      ),
      'Max': min_max_and_count.max,
      'Mean': mean_and_var.mean,
      'Min': min_max_and_count.min,
      'Standard Deviation': mean_and_var.stddev,
      'Variance': mean_and_var.var,
  }
