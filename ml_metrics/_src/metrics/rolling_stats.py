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
"""Individual statistics metrics."""

from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.aggregates import types


_METRIC_PYDOC_POSTFIX = """

  The metric is computed based on non-nan values within the batch.

  Args:
    batch: A batch of numbers.

  Returns:
    Metric value.
"""


def var(batch: types.NumbersT) -> float:
  """Computes the variance in a batch."""
  return rolling_stats.MeanAndVariance().add(batch).var


var.__doc__ += _METRIC_PYDOC_POSTFIX


def stddev(batch: types.NumbersT) -> float:
  """Computes the standard deviation in a batch."""
  return rolling_stats.MeanAndVariance().add(batch).stddev


stddev.__doc__ += _METRIC_PYDOC_POSTFIX


def mean(batch: types.NumbersT) -> float:
  """Computes the mean in a batch."""
  return rolling_stats.MeanAndVariance().add(batch).mean


mean.__doc__ += _METRIC_PYDOC_POSTFIX


def count(batch: types.NumbersT) -> int:
  """Computes the number of elements in a batch."""
  return rolling_stats.MeanAndVariance().add(batch).count


count.__doc__ += _METRIC_PYDOC_POSTFIX


def total(batch: types.NumbersT) -> float:
  """Computes the total sum of a batch."""
  return rolling_stats.MeanAndVariance().add(batch).total


total.__doc__ += _METRIC_PYDOC_POSTFIX
