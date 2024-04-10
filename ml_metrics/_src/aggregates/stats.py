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
from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import types
import numpy as np


@dataclasses.dataclass(kw_only=True)
class StatsState(base.MergeableMetric):
  """State of a statistics aggregation."""

  batch_score_fn: Callable[..., types.NumbersT] | None = None
  _min: types.NumbersT | None = None
  _max: types.NumbersT | None = None
  _count: int = 0
  _mean: types.NumbersT | None = None
  _var: types.NumbersT | None = None

  def add(self, batch: types.NumbersT) -> 'StatsState':
    """Update the statistics with the given batch.

    Args:
      batch:
        A non-vacant series of numerical values.

    Returns:
      StatsState

    Raise:
      ValueError:
        If the `batch` is empty.
    """

    if not list(batch):
      raise ValueError('`batch` must not be empty.')

    if not self._count:
      # This assumes the first dimension is the batch dimension.
      if self.batch_score_fn is not None:
        batch = self.batch_score_fn(batch)
      # Mininums and maximums.
      self._min = np.min(batch, axis=0)
      self._max = np.max(batch, axis=0)
      # Sufficient statistics for Mean, variance and standard deviation.
      self._count = len(batch)
      self._mean = np.mean(batch, axis=0)
      self._var = np.var(batch, axis=0)
      return self.result()

    batch_state = StatsState(batch_score_fn=self.batch_score_fn)
    batch_state.add(batch)
    self.merge(batch_state)
    return batch_state.result()

  @property
  def min(self) -> types.NumbersT:
    return self._min

  @property
  def max(self) -> types.NumbersT:
    return self._max

  @property
  def var(self) -> types.NumbersT:
    return self._var

  @property
  def stddev(self) -> types.NumbersT:
    return self._var and np.sqrt(self._var)

  @property
  def mean(self) -> types.NumbersT:
    return self._mean

  @property
  def count(self) -> types.NumbersT:
    return self._count

  @property
  def total(self) -> types.NumbersT:
    return self._mean * self._count if self._count > 0 else 0.0

  def merge(self, other: 'StatsState'):
    if not self._count or not other.count:
      raise ValueError('Both StatsStates must not be empty.')

    self._min = min(self._min, other.min)
    self._max = max(self._max, other.max)

    prev_mean, prev_count = self._mean, self._count
    self._count += other.count
    self._mean += (other.mean - prev_mean) * (other.count / self._count)
    # Reference
    # (https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups)
    prev_count_ratio = prev_count / self._count
    other_count_ratio = other.count / self._count
    self._var = (
        prev_count_ratio * self._var
        + other_count_ratio * other.var
        + prev_count_ratio * (prev_mean - self._mean) ** 2
        + other_count_ratio * (other.mean - self._mean) ** 2
    )

  def result(self) -> 'StatsState':
    return self
