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

"""Common statistics aggregations."""

from collections.abc import Callable
import dataclasses
from ml_metrics._src.aggregates import types
import numpy as np


@dataclasses.dataclass(kw_only=True)
# TODO: b/311207032 - implements MergeableMetric interface.
class StatsState:
  """State of a statistics aggregation."""

  batch_score_fn: Callable[..., types.NumbersT] | None = None
  _min: types.NumbersT | None = None
  _max: types.NumbersT | None = None
  _count: int = 0
  _mean: types.NumbersT | None = None
  _var_sum: types.NumbersT | None = None
  _diff_sum: types.NumbersT | None = None

  def add(self, batch: types.NumbersT) -> None:
    """Update the statistics with the given batch."""
    # This assumes the first dimension is the batch dimension.
    if self.batch_score_fn is not None:
      batch = self.batch_score_fn(batch)
    # Mininums and maximums.
    if self._min is None:
      self._min = np.min(batch, axis=0)
    else:
      self._min = np.minimum(self._min, np.min(batch, axis=0))
    if self._max is None:
      self._max = np.max(batch, axis=0)
    else:
      self._max = np.maximum(self._max, np.max(batch, axis=0))
    # Sufficient statistics for Mean, variance and standard deviation.
    if not self._count:
      self._count = len(batch)
      self._mean = np.mean(batch, axis=0)
      self._diff_sum = np.sum(batch - self._mean, axis=0)
      self._var_sum = np.sum(np.power(batch - self._mean, 2), axis=0)
    else:
      prev_mean, prev_count = self._mean, self._count
      self._count += len(batch)
      batch_sum = np.sum(batch, axis=0)
      self._mean += (batch_sum - prev_mean * len(batch)) / self._count
      mean_delta = self._mean - prev_mean
      prev_diff_sum = self._diff_sum
      self._diff_sum += -mean_delta * prev_count + np.sum(
          batch - self._mean, axis=0
      )
      batch_var_sum = np.sum(np.power(batch - self._mean, 2), axis=0)
      self._var_sum += (
          -2 * prev_diff_sum + mean_delta**2 * prev_count + batch_var_sum
      )

  @property
  def min(self) -> types.NumbersT:
    return self._min

  @property
  def max(self) -> types.NumbersT:
    return self._max

  @property
  def var(self) -> types.NumbersT:
    return self._var_sum and self._var_sum / self._count

  @property
  def stddev(self) -> types.NumbersT:
    return self._var_sum and np.sqrt(self.var)

  @property
  def mean(self) -> types.NumbersT:
    return self._mean

  @property
  def count(self) -> types.NumbersT:
    return self._count

  @property
  def total(self) -> types.NumbersT:
    return self._mean * self._count if self._count > 0 else 0.0
