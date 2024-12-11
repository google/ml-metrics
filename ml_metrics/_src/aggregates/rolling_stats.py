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

import abc
import collections
from collections.abc import Callable
import dataclasses
import math
from typing import Any, Self

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import types
from ml_metrics._src.utils import math_utils
import numpy as np

_FLOAT_EPSNEG = np.finfo(float).epsneg


@dataclasses.dataclass(slots=True)
class FixedSizeSample(base.MergeableMetric):
  """Generates a fixed size sample of the data stream.

  This sampler is used to generate a fixed size sample of the data stream.
  Initially, the reservoir of datastream values will grow to the size of 'size'.
  Then, subsequent values may replace values in the reservoir, depending on
  random sampling.

  This is also known as Reservoir Sampling:
  https://en.wikipedia.org/wiki/Reservoir_sampling.

  Attributes:
    max_size: The number of samples to store in the reservoir.
    _random_seed: The random_seed used to initialize the random number
      generator.
    _reservoir: The reservoir of samples. The length of this reservoir will
      always be less than or equal to "max_size".
    _num_samples_reviewed: The running number of samples reviewed. This counts
      both samples that were and samples that were not added to the reservoir.
  """

  max_size: int
  _random_seed: dataclasses.InitVar[int | None] = None
  _reservoir: list[Any] = dataclasses.field(default_factory=list)
  _num_samples_reviewed: int = 0

  def __post_init__(self, seed):
    self._rng = np.random.default_rng(seed)
    self._w = np.exp(
        np.log(self._rng.uniform(low=_FLOAT_EPSNEG)) / self.max_size
    )

  def _add_samples_to_reservoir(self, samples: list[Any], n: int):
    # This is Algorithm L: https://dl.acm.org/doi/10.1145/198429.198435.

    i = -1
    for _ in range(min(self.max_size - len(self._reservoir), n)):
      i += 1
      self._reservoir.append(samples[i])

    while i < n:
      i += (
          np.floor(
              np.log(self._rng.uniform(low=_FLOAT_EPSNEG)) / np.log(1 - self._w)
          ).astype(int)
          + 1
      )

      if i < n:
        self._reservoir[self._rng.integers(self.max_size)] = samples[i]
        self._w *= np.exp(
            np.log(self._rng.uniform(low=_FLOAT_EPSNEG)) / self.max_size
        )

  def _merge_reservoirs(
      self,
      reservoir_new: list[Any],
      num_samples_new: int,
  ) -> list[Any]:
    if (
        len(combined_reservoir := self._reservoir + reservoir_new)
        <= self.max_size
    ):
      # If the combined reservoir is within the size limit, we can simply make
      # it our new reservoir.
      return combined_reservoir

    else:

      # TODO: b/370053191 - For efficiency, sample from the combined reservoir
      # in one-shot.
      merged_res = []
      num_samples_orig = self._num_samples_reviewed
      for _ in range(self.max_size):
        from_orig = self._rng.uniform() < (
            num_samples_orig / (num_samples_orig + num_samples_new)
        )

        if from_orig:
          merged_res.append(
              self._reservoir.pop(self._rng.integers(len(self._reservoir)))
          )
          num_samples_orig -= 1
        else:
          merged_res.append(
              reservoir_new.pop(self._rng.integers(len(reservoir_new)))
          )
          num_samples_new -= 1

    return merged_res

  def add(self, inputs: types.NumbersT) -> 'FixedSizeSample':
    num_inputs = len(inputs)

    self._add_samples_to_reservoir(samples=inputs, n=num_inputs)

    self._num_samples_reviewed += num_inputs

    return self

  def merge(self, other: 'FixedSizeSample') -> 'FixedSizeSample':
    num_new_samples_reviewed = other._num_samples_reviewed  # pylint: disable=protected-access

    self._reservoir = self._merge_reservoirs(
        reservoir_new=other.result(),
        num_samples_new=num_new_samples_reviewed,
    )

    self._num_samples_reviewed += num_new_samples_reviewed

    return self

  def result(self) -> types.NumbersT:
    return self._reservoir


HistogramResult = collections.namedtuple(
    'HistogramResult', ('hist', 'bin_edges'),
)


@dataclasses.dataclass(slots=True)
class Histogram(base.MergeableMetric):
  """Computes the Histogram of the inputs.

  Attributes:
    range: The lower and upper range of the bins. e.g. range = (0, 1).
    bins: The number of buckets to use.
    _hist: The values of the histogram.
    _bin_edges: The bin edges of the histogram. All but the right-most bin are
      half-open. I.e. if the bins_edges are (0, 1, 2, 3, ..., 8, 9, 10), then
      the bin ranges are [0, 1), [1, 2), [2, 3), ... [8, 9), [9, 10].
  """

  range: tuple[float, float]
  bins: int = 10
  _hist: np.ndarray = dataclasses.field(init=False)
  _bin_edges: np.ndarray = dataclasses.field(init=False)

  def __post_init__(self):
    self._hist, self._bin_edges = np.histogram(
        a=(), bins=self.bins, range=self.range
    )

  @property
  def hist(self) -> np.ndarray:
    return self._hist

  @property
  def bin_edges(self) -> np.ndarray:
    return self._bin_edges

  def _merge(self, hist: np.ndarray, bin_edges: np.ndarray) -> 'Histogram':
    if not np.array_equal(bin_edges, self._bin_edges):

      # Self hist and new hist have different bin edges.
      if self._bin_edges.shape == bin_edges.shape:
        # Self hist and new hist have the same shape of bin edges.
        raise ValueError(
            'The bin edges of the two Histograms must be equal, but recieved'
            f' self._bin_edges={self._bin_edges} and new_bin_edges={bin_edges}'
            ' which have different elements at indices'
            f' {np.where(self._bin_edges != bin_edges)}.'
        )

      else:
        # Self hist and new hist have different shapes of bin edges.
        raise ValueError(
            'The bin edges of the two Histograms must be equal, but recieved'
            f' self._bin_edges={self._bin_edges} and new_bin_edges={bin_edges}'
            f' which have shapes {self._bin_edges.shape} and {bin_edges.shape},'
            ' respectively.'
        )

    self._hist = self._hist + hist
    return self

  def add(
      self, inputs: types.NumbersT, weights: types.NumbersT | None = None
  ) -> 'Histogram':
    new_histogram, new_bin_edges = np.histogram(
        inputs,
        bins=self.bins,
        range=self.range,
        weights=weights,
    )
    return self._merge(new_histogram, new_bin_edges)

  def merge(self, other: 'Histogram') -> 'Histogram':
    return self._merge(other.hist, other.bin_edges)

  def result(self) -> HistogramResult:
    return HistogramResult(
        hist=self._hist.copy(), bin_edges=self._bin_edges.copy()
    )


@dataclasses.dataclass(kw_only=True, eq=True)
class MeanAndVariance(base.MergeableMetric, base.CallableMetric):
  """Computes the mean and variance of a batch of values."""

  # TODO(b/345249574): (1) Introduce StatsEnum to indicate the metrics to be
  # computed. (2) Remove score_batch_fn.

  batch_score_fn: Callable[..., types.NumbersT] | None = None
  _count: types.NumbersT = 0
  _mean: types.NumbersT = np.nan
  _var: types.NumbersT = np.nan

  def as_agg_fn(
      self,
      *,
      nested: bool = False,
  ) -> base.AggregateFn:
    return base.as_agg_fn(
        self.__class__,
        batch_score_fn=self.batch_score_fn if not nested else None,
        nested=nested,
        agg_preprocess_fn=self.batch_score_fn if nested else None,
    )

  def add(self, batch: types.NumbersT) -> Self:
    """Update the statistics with the given batch.

    If `batch_score_fn` is provided, it will evaluate the batch and assign a
    score to each item. Subsequently, the statistics are computed based on
    non-nan values within the batch. If a certain dimension in batch is all nan,
    mean and variance corresponding to that dimension will be nan, count for
    that dimension will be 0.

    Args:
      batch: A non-vacant series of values.

    Returns:
      MeanAndVariance

    Raise:
      ValueError:
        If `batch_score_fn` is provided and the returned series is not the same
        length as the `batch`.
    """

    if np.all(self.count == 0):
      # This assumes the first dimension is the batch dimension.
      if self.batch_score_fn is not None:
        org_batch_size = len(batch)
        batch = self.batch_score_fn(batch)
        if len(batch) != org_batch_size:
          raise ValueError(
              'The `batch_score_fn` must return a series of the same length as'
              ' the `batch`.'
          )

      # Sufficient statistics for mean, variance and standard deviation.
      self._count = np.nansum(~np.isnan(batch), axis=0)
      self._mean = np.nanmean(batch, axis=0)
      self._var = np.nanvar(batch, axis=0)
      return self.result()

    batch_state = MeanAndVariance(batch_score_fn=self.batch_score_fn)
    batch_state.add(batch)
    self.merge(batch_state)
    return batch_state.result()

  @property
  def var(self) -> types.NumbersT:
    return self._var

  @property
  def stddev(self) -> types.NumbersT:
    return np.sqrt(self._var)

  @property
  def mean(self) -> types.NumbersT:
    return self._mean

  @property
  def count(self) -> types.NumbersT:
    return self._count

  @property
  def total(self) -> types.NumbersT:
    return np.where(
        self._count == 0, np.zeros_like(self._mean), self._mean * self._count
    )

  def merge(self, other: 'MeanAndVariance'):
    if np.all(other.count == 0):
      return

    # When self count is zero and other's count is not zero, we copy the state
    # from other. The values and and shape of the states will be from other.
    if np.all(self.count == 0):
      self._count = other.count
      self._mean = other.mean
      self._var = other.var
      return

    prev_mean, prev_count = np.copy(self._mean), np.copy(self._count)
    self._count += other.count
    self._mean += (other.mean - prev_mean) * math_utils.safe_divide(
        other.count, self._count
    )

    # Reference
    # (https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups)
    prev_count_ratio = math_utils.safe_divide(prev_count, self._count)
    other_count_ratio = math_utils.safe_divide(other.count, self._count)
    self._var = (
        prev_count_ratio * self._var
        + other_count_ratio * other.var
        + prev_count_ratio * (prev_mean - self._mean) ** 2
        + other_count_ratio * (other.mean - self._mean) ** 2
    )

  def result(self) -> 'MeanAndVariance':
    return self

  def __str__(self):
    return (
        f'count: {self.count}, total: {self.total}, mean: {self.mean}, '
        f'var: {self.var}, stddev: {self.stddev}'
    )


def MeanAndVarianceAggFn(*args, **kwargs):  # pylint: disable=invalid-name
  """AggFn wrapper of the MeanAndVariance."""
  return MeanAndVariance(*args, **kwargs).as_agg_fn()


# TODO(b/345249574): Add a preprocessing function of len per row.
@dataclasses.dataclass(slots=True)
class MinMaxAndCount(base.MergeableMetric):
  """Computes the Min, Max, and Count.

  Given a batch of inputs, MinMaxAndCount computes the following statistics:
    count: The total the number of input values across all the batches.
    min: The number of inputs in the batch that has the least number of inputs.
    max: The number of inputs in the batch that has the most number of inputs.
  """

  batch_score_fn: Callable[..., types.NumbersT] | None = None
  axis: int | None = None
  _count: int = 0
  _min: int = np.inf
  _max: int = 0

  def __eq__(self, other: 'MinMaxAndCount') -> bool:
    return (
        self._count == other.count
        and self._min == other.min
        and self._max == other.max
    )

  @property
  def count(self) -> int:
    return self._count

  @property
  def min(self) -> int:
    return self._min

  @property
  def max(self) -> int:
    return self._max

  def add(self, inputs: types.NumbersT) -> 'MinMaxAndCount':
    self._count += np.asarray(inputs).size

    if self.batch_score_fn is not None:
      inputs = self.batch_score_fn(inputs)

    self._min = np.minimum(self._min, np.min(inputs, axis=self.axis))
    self._max = np.maximum(self._max, np.max(inputs, axis=self.axis))

    return self

  def merge(self, other: 'MinMaxAndCount') -> 'MinMaxAndCount':
    self._count += other.count
    self._min = np.min((self._min, other.min), axis=self.axis)
    self._max = np.max((self._max, other.max), axis=self.axis)

    return self

  def result(self) -> 'MinMaxAndCount':
    return self


@dataclasses.dataclass(slots=True)
class _R2TjurBase(abc.ABC, base.MergeableMetric):
  """Base class for Tjur's R^2.

  Also known as Tjur's D or Tjur's coefficient of discrimination, the Tjur
  pseudo R^2 value compares the average fitted probability π¯ of the two
  response outcomes. In particular, it is the difference between the average
  fitted probability for the binary outcome coded to 1 (success level) and the
  average fitted probability for the binary outcome coded to 0 (the failure
  level).


  https://www.statease.com/docs/v12/contents/advanced-topics/glm/tjur-pseudo-r-squared/

  sum_y_true: The sum of y_true.
  sum_y_pred: The sum of y_true * y_pred.
  sum_neg_y_true: The sum of 1 - y_true.
  sum_neg_y_pred: The sum of (1 - y_true) * y_pred.
  """

  sum_y_true: float = 0
  sum_y_pred: float = 0
  sum_neg_y_true: float = 0
  sum_neg_y_pred: float = 0

  def __eq__(self, other: '_R2TjurBase') -> bool:
    return (
        self.sum_y_true == other.sum_y_true
        and self.sum_y_pred == other.sum_y_pred
        and self.sum_neg_y_true == other.sum_neg_y_true
        and self.sum_neg_y_pred == other.sum_neg_y_pred
    )

  def add(
      self, y_true: types.NumbersT, y_pred: types.NumbersT
  ) -> '_R2TjurBase':
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    neg_y_true = 1 - y_true

    self.sum_y_true += np.sum(y_true)
    self.sum_y_pred += np.sum(y_true * y_pred)
    self.sum_neg_y_true += np.sum(neg_y_true)
    self.sum_neg_y_pred += np.sum(neg_y_true * y_pred)

    return self

  def merge(self, other: '_R2TjurBase') -> '_R2TjurBase':
    self.sum_y_true += other.sum_y_true
    self.sum_y_pred += other.sum_y_pred
    self.sum_neg_y_true += other.sum_neg_y_true
    self.sum_neg_y_pred += other.sum_neg_y_pred

    return self

  @abc.abstractmethod
  def result(self) -> types.NumbersT:
    """Must be overwritten by the specific Tjur's R^2 Metric."""
    pass


class R2Tjur(_R2TjurBase):

  def result(self) -> types.NumbersT:
    if math.isclose(self.sum_y_true, 0) or math.isclose(self.sum_neg_y_true, 0):
      return float('nan')

    return (
        self.sum_y_pred / self.sum_y_true
        - self.sum_neg_y_pred / self.sum_neg_y_true
    )


class R2TjurRelative(_R2TjurBase):

  def result(self) -> types.NumbersT:
    if math.isclose(self.sum_y_true, 0) or math.isclose(self.sum_neg_y_pred, 0):
      return float('nan')

    return (
        self.sum_y_pred
        * self.sum_neg_y_true
        / self.sum_y_true
        / self.sum_neg_y_pred
    )


@dataclasses.dataclass(slots=True)
class RRegression(base.MergeableMetric):
  """Computes the Pearson Correlation Coefficient (PCC).

  The Pearson correlation coefficient (PCC) is a correlation coefficient that
  measures linear correlation between two sets of data. It is the ratio between
  the covariance of two variables and the product of their standard deviations;
  thus, it is essentially a normalized measurement of the covariance, such that
  the result always has a value between -1 and 1. As with covariance itself, the
  measure can only reflect a linear correlation of variables, and ignores many
  other types of relationships or correlations. As a simple example, one would
  expect the age and height of a sample of teenagers from a high school to have
  a Pearson correlation coefficient significantly greater than 0, but less than
  1 (as 1 would represent an unrealistically perfect correlation).

  https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
  """

  # If True, center the data matrix x and the target vector y.
  # The centered r-regression is the "Pearson's Correlation".
  # The not-centered r-regression is the "Reflective Correlation".
  center: bool = True

  num_samples: int = 0
  sum_x: types.NumbersT = 0
  sum_y: float = 0
  sum_xx: types.NumbersT = 0  # sum(x**2)
  sum_yy: float = 0  # sum(y**2)
  sum_xy: types.NumbersT = 0  # sum(x * y)

  def __eq__(self, other: 'RRegression') -> bool:
    return (
        self.num_samples == other.num_samples
        and self.sum_x == other.sum_x
        and self.sum_y == other.sum_y
        and self.sum_xx == other.sum_xx
        and self.sum_yy == other.sum_yy
        and self.sum_xy == other.sum_xy
    )

  def add(self, x: types.NumbersT, y: types.NumbersT) -> 'RRegression':
    """Updates the Class with the given batch.

    Args:
      x: The data matrix of shape (n_samples, n_examples).
      y: The target vector of shape (n_samples,).

    Returns:
      The updated Class.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    self.num_samples += len(y)
    self.sum_x += np.sum(x, axis=0)
    self.sum_y += np.sum(y)
    self.sum_xx += np.sum(x**2, axis=0)
    self.sum_yy += np.sum(y**2)
    self.sum_xy += np.sum(x * (y if x.ndim == 1 else y[:, np.newaxis]), axis=0)

    return self

  def merge(self, other: 'RRegression') -> 'RRegression':
    self.num_samples += other.num_samples
    self.sum_x += other.sum_x
    self.sum_y += other.sum_y
    self.sum_xx += other.sum_xx
    self.sum_yy += other.sum_yy
    self.sum_xy += other.sum_xy

    return self

  def result(self) -> types.NumbersT:
    """Calculates the Pearson Correlation Coefficient (PCC).

    PCC = cov(X, Y) / std(X) / std(Y)
    where cov is the covariance, and std(X) is the standard deviation of X;
    and analagously for std(Y).

    After substituting estimates of the covariances and variances
    PCC = sum((x_i - x_bar) * (y_i - y_bar))
          / sqrt(sum((x_i - x_bar)**2)) / sqrt(sum((x_i - x_bar)**2))
    where x_i and y_i are the individual sampled points indexed with i, and
    x_bar is the sample mean; and analagously for y_bar.

    Rearranging the PCC formula gives us
    PCC = (n * sum(x_i * y_i) - sum(x_i) * sum(y_i))
          / sqrt(n * sum(x_i ** 2) - sum(x_i)**2)
          / sqrt(n * sum(y_i ** 2) - sum(y_i)**2)
    where n is sample size, and x_i and y_i are defined as above.

    Simplifying this yields
    PCC = (sum(x_i * y_i) - n * x_bar * y_bar)
          / sqrt(sum(x_i ** 2) - n * x_bar ** 2)
          / sqrt(sum(y_i ** 2) - n * y_bar ** 2)
    where n, x_i, y_i, x_bar, and y_bar are defined as above.

    Returns:
      The Pearson Correlation Coefficient.
    """
    if self.center:  # Pearson's Correlation
      # numerator = sum(x_i * y_i) - n * x_bar * y_bar

      # denominator_x = sqrt(sum(x_i ** 2) - n * x_bar ** 2)
      # denominator_y = sqrt(sum(y_i ** 2) - n * y_bar ** 2)
      # denominator = denominator_x * denominator_y

      numerator = self.sum_xy - self.sum_x * self.sum_y / self.num_samples

      denominator_x = np.sqrt(self.sum_xx - self.sum_x**2 / self.num_samples)
      denominator_y = np.sqrt(self.sum_yy - self.sum_y**2 / self.num_samples)
      denominator = denominator_x * denominator_y

    else:  # Reflective Correlation
      numerator = self.sum_xy
      denominator = np.sqrt(self.sum_xx * self.sum_yy)

    return numerator / denominator


@dataclasses.dataclass(slots=True)
class SymmetricPredictionDifference(base.MergeableMetric):
  """Computes the Symmetric Prediction Difference.

  Creates a summary model by taking the pointwise symmetric relative prediction
  difference between two input values. For two input values x and y, the
  pointwise symmetric relative prediction difference is defined as
  2 * |x - y| / |x + y|.
  """

  num_samples: int = 0
  sum_half_pointwise_rel_diff: float = 0
  # TODO: b/356933410 - Add k_epsilon.

  def add(
      self, x: types.NumbersT, y: types.NumbersT
  ) -> 'SymmetricPredictionDifference':
    x = np.asarray(x).astype('float64')
    y = np.asarray(y).astype('float64')

    if x.shape != y.shape:
      raise ValueError(
          'SymmetricPredictionDifference.add() requires x and y to have the'
          f' same shape, but recieved x={x} and y={y} with x.shape={x.shape}'
          f' and y.shape={y.shape}'
      )

    self.num_samples += x.size

    # TODO: b/356933410 - Add logic for k_epsilon.
    self.sum_half_pointwise_rel_diff += np.sum(
        math_utils.safe_divide(np.abs(x - y), np.abs(x + y))
    )

    return self

  def merge(
      self, other: 'SymmetricPredictionDifference'
  ) -> 'SymmetricPredictionDifference':
    self.num_samples += other.num_samples
    self.sum_half_pointwise_rel_diff += other.sum_half_pointwise_rel_diff

    return self

  def result(self) -> float:
    if self.num_samples == 0:
      return float('nan')

    return 2 * self.sum_half_pointwise_rel_diff / self.num_samples
