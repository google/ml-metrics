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
from __future__ import annotations

import abc
import collections
from collections.abc import Callable, Iterable
import dataclasses
import math
from typing import Any, Generic, Self, TypeVar

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import types
from ml_metrics._src.utils import math_utils
import numpy as np

_EPSNEG = np.finfo(float).epsneg
_T = TypeVar('_T')


@dataclasses.dataclass(kw_only=True)
class UnboundedSampler(base.CallableMetric, base.HasAsAggFn):
  """Stores all the inputs in memory."""

  _samples: tuple[list[Any], ...] = ()
  _multi_input: bool = True

  @property
  def samples(self) -> tuple[list[Any], ...]:
    return self._samples

  @property
  def multi_input(self) -> bool:
    return self._multi_input

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__)

  def new(self, *inputs: tuple[Iterable[Any], ...]) -> Self:
    multi_input = len(inputs) != 1
    samples = tuple(list(input_) for input_ in inputs)
    return self.__class__(_samples=samples, _multi_input=multi_input)

  def merge(self, other: Self) -> Self:
    if not self._samples:
      self._samples = tuple([] for _ in other.samples)
      self._multi_input = other.multi_input
    for samples, others in zip(self._samples, other.samples, strict=True):
      samples.extend(others)
    return self

  def result(self) -> tuple[list[Any], ...] | list[Any]:
    if self._multi_input:
      return self._samples
    return self._samples[0]


@dataclasses.dataclass(slots=True)
class FixedSizeSample(base.MergeableMetric, base.HasAsAggFn):
  """Generates a fixed size sample of the data stream.

  This sampler is used to generate a fixed size sample of the data stream.
  Initially, the reservoir of datastream values will grow to the size of 'size'.
  Then, subsequent values may replace values in the reservoir, depending on
  random sampling.

  This is also known as Reservoir Sampling:
  https://en.wikipedia.org/wiki/Reservoir_sampling.

  Attributes:
    max_size: The number of samples to store in the reservoir.
    seed: The random_seed used to initialize the random number generator.
    reservoir: The reservoir of samples. The length of this reservoir will
      always be less than or equal to "max_size".
    num_samples_reviewed: The running number of samples reviewed. This counts
      both samples that were and samples that were not added to the reservoir.
  """

  max_size: int
  seed: int | None = dataclasses.field(default=None, kw_only=True)
  _reservoir: list[Any] = dataclasses.field(default_factory=list)
  _num_samples_reviewed: int = 0
  _logw: float = dataclasses.field(init=False)
  _rng: np.random.Generator = dataclasses.field(init=False)

  def __post_init__(self):
    self._rng = np.random.default_rng(self.seed)
    self._logw = np.log(self._rng.uniform(low=_EPSNEG)) / self.max_size

  @property
  def num_samples_reviewed(self) -> int:
    return self._num_samples_reviewed

  @property
  def reservoir(self) -> list[Any]:
    return self._reservoir

  @property
  def logw(self) -> float:
    return self._logw

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__, self.max_size, seed=self.seed)

  def _add_samples_to_reservoir(self, samples: list[Any], n: int):
    # This is Algorithm L: https://dl.acm.org/doi/10.1145/198429.198435.
    len_n = min(self.max_size - len(self._reservoir), n)
    self._reservoir.extend(samples[:len_n])
    i = len_n - 1
    while i < n:
      w = np.exp(self._logw)
      di = np.log(self._rng.uniform(low=_EPSNEG)) / np.log(1 - w)
      i += np.floor(di).astype(int) + 1
      if i < n:
        self._reservoir[self._rng.integers(self.max_size)] = samples[i]
        self._logw += np.log(self._rng.uniform(low=_EPSNEG)) / self.max_size
    self._num_samples_reviewed += n

  def _merge_reservoirs(self, other: FixedSizeSample) -> list[Any]:
    # TODO: b/370053191 - For efficiency, sample from the combined reservoir
    # in one-shot.
    result = []
    num_samples_orig = self._num_samples_reviewed
    reservoir_new, num_samples_new = other.reservoir, other.num_samples_reviewed
    while len(result) < self.max_size and num_samples_orig + num_samples_new:
      thr_from_orig = num_samples_orig / (num_samples_orig + num_samples_new)
      if self._rng.uniform() < thr_from_orig:
        sample = self._reservoir.pop(self._rng.integers(len(self._reservoir)))
        num_samples_orig -= 1
      else:
        sample = reservoir_new.pop(self._rng.integers(len(reservoir_new)))
        num_samples_new -= 1
      result.append(sample)
    return result

  def add(self, inputs: types.NumbersT):
    self._add_samples_to_reservoir(inputs, n=len(inputs))

  def merge(self, other: FixedSizeSample):
    if self.seed != other.seed:
      raise ValueError(
          'The seeds of the two samplers must be equal, but recieved'
          f' self.seed={self.seed} and other.seed={other.seed}.'
      )
    self._reservoir = self._merge_reservoirs(other)
    self._num_samples_reviewed += other.num_samples_reviewed
    self._logw += other.logw

  def result(self) -> types.NumbersT:
    return self._reservoir


HistogramResult = collections.namedtuple(
    'HistogramResult', ('hist', 'bin_edges'),
)


@dataclasses.dataclass(slots=True, kw_only=True)
class Histogram(base.CallableMetric, base.HasAsAggFn):
  """Computes the Histogram of the inputs.

  Attributes:
    range: The lower and upper range of the bins. e.g. range = (0, 1).
    bins: The number of buckets to use.
    _hist: The values of the histogram.
    _bin_edges: The bin edges of the histogram. All but the right-most bin are
      half-open. I.e. if the bins_edges are (0, 1, 2, 3, ..., 8, 9, 10), then
      the bin ranges are [0, 1), [1, 2), [2, 3), ... [8, 9), [9, 10].
  """

  range: tuple[float, float] | None = None
  bins: int | Iterable[float] = dataclasses.field(default=10, kw_only=True)
  _hist: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0))
  _bin_edges: np.ndarray = dataclasses.field(
      default_factory=lambda: np.empty(0)
  )

  def __post_init__(self):
    if not isinstance(self.bins, int):
      self.bins = tuple(self.bins)
    if not self._hist.size and not self._bin_edges.size:
      self._hist, self._bin_edges = np.histogram(
          a=(), bins=self.bins, range=self.range
      )
      return

  @property
  def hist(self) -> np.ndarray:
    return self._hist

  @property
  def bin_edges(self) -> np.ndarray:
    return self._bin_edges

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__, range=self.range, bins=self.bins)

  def new(
      self, inputs: types.NumbersT, weights: types.NumbersT | None = None
  ) -> Histogram:
    new_histogram, new_bin_edges = np.histogram(
        inputs,
        bins=self.bins,
        range=self.range,
        weights=weights,
    )
    return self.__class__(
        range=self.range,
        bins=self.bins,
        _hist=new_histogram,
        _bin_edges=new_bin_edges,
    )

  def merge(self, other: Self) -> Self:
    hist, bin_edges = other.hist, other.bin_edges
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

  def result(self) -> HistogramResult:
    return HistogramResult(
        hist=self._hist.copy(), bin_edges=self._bin_edges.copy()
    )


@dataclasses.dataclass(slots=True, kw_only=True)
class Counter(base.CallableMetric, base.HasAsAggFn, Generic[_T]):
  """An CallableMetric version of collections.Counter."""

  _counter: collections.Counter[_T] = dataclasses.field(
      default_factory=collections.Counter
  )

  @property
  def counter(self):
    return self._counter

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__)

  def new(self, inputs: Iterable[_T]) -> Self:
    return self.__class__(_counter=collections.Counter(inputs))

  def merge(self, other: Self) -> Self:
    self._counter.update(other.counter)
    return self

  def result(self) -> collections.Counter[_T]:
    return self._counter


@dataclasses.dataclass(slots=True, kw_only=True)
class Count(base.CallableMetric):
  """Computes the count of a batch of values."""

  batch_score_fn: Callable[..., types.NumbersT] | None = None
  _count: types.NumbersT = 0

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

  @property
  def count(self) -> types.NumbersT:
    return self._count

  def new(self, *batch: types.NumbersT) -> types.NumbersT:
    """Computes the sufficient statistics of a batch of values."""
    if not self.batch_score_fn and len(batch) > 1:
      raise ValueError(
          'Multi-column inputs requires a batch_score_fn to convert it to a'
          f' single column, but received {len(batch)} columns.'
      )
    batch = self.batch_score_fn(*batch) if self.batch_score_fn else batch[0]
    return self.__class__(_count=len(batch))

  def merge(self, other: Self):
    self._count += other.count

  def result(self) -> Self:
    return self.count

  def __str__(self):
    return f'count: {self.count}'


@dataclasses.dataclass(kw_only=True, eq=True)
class Mean(Count):
  """Computes the mean and variance of a batch of values."""

  _mean: types.NumbersT = np.nan
  _input_shape: tuple[int, ...] = ()

  def new(self, batch: types.NumbersT) -> types.NumbersT:
    """Computes the sufficient statistics of a batch of values.

    If `batch_score_fn` is provided, it will evaluate the batch and assign a
    score to each item. Subsequently, the statistics are computed based on
    non-nan values within the batch. If a certain dimension in batch is all nan,
    mean and variance corresponding to that dimension will be nan, count for
    that dimension will be 0.

    Args:
      batch: A non-vacant series of values.

    Returns:
      Mean
    """
    batch = np.asarray(
        self.batch_score_fn(batch) if self.batch_score_fn else batch
    )
    return self.__class__(
        _count=np.sum(~np.isnan(batch), axis=0),
        _mean=np.nanmean(batch, axis=0),
        _input_shape=batch.shape if batch.size else (),
    )

  @property
  def mean(self) -> types.NumbersT:
    return self._mean

  @property
  def total(self) -> types.NumbersT:
    return math_utils.where(self._count > 0, self._mean * self._count, 0)

  @property
  def input_shape(self) -> tuple[int, ...]:
    return self._input_shape

  def merge(self, other: Mean):
    if np.all(np.isnan(other.mean)):
      return
    self._input_shape = self._input_shape or other.input_shape
    if other.input_shape and other.input_shape[1:] != self._input_shape[1:]:
      raise ValueError(
          f'Incompatible shape {other.input_shape} while the'
          f' other have shape {self._input_shape}.'
      )
    self._count += other.count
    mean_diff = math_utils.nanadd(other.mean, -self._mean)
    update = mean_diff * math_utils.safe_divide(other.count, self._count)
    self._mean = math_utils.nanadd(self._mean, update)

  def result(self) -> Self:
    return self.mean

  def __str__(self):
    return f'mean: {self.mean}'


@dataclasses.dataclass(kw_only=True, eq=True)
class MeanAndVariance(Mean):
  """Computes the mean and variance of a batch of values."""

  _var: types.NumbersT = np.nan

  def new(self, batch: types.NumbersT) -> types.NumbersT:
    batch = np.asarray(
        self.batch_score_fn(batch) if self.batch_score_fn else batch
    )
    return self.__class__(
        _count=np.sum(~np.isnan(batch), axis=0),
        _mean=np.nanmean(batch, axis=0),
        _var=np.nanvar(batch, axis=0),
        _input_shape=batch.shape if batch.size else (),
    )

  @property
  def var(self) -> types.NumbersT:
    return self._var

  @property
  def stddev(self) -> types.NumbersT:
    return np.sqrt(self._var)

  def merge(self, other: MeanAndVariance):
    if np.all(np.isnan(other.var)):
      return
    prev_mean, prev_count = np.copy(self._mean), np.copy(self._count)
    super().merge(other)
    if np.all(np.isnan(self._var)):
      self._var = other.var
      return
    # Reference
    # (https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups)
    prev_count_ratio = math_utils.safe_divide(prev_count, self._count)
    other_count_ratio = math_utils.safe_divide(other.count, self._count)
    delta_mean = math_utils.nanadd(self._mean, -prev_mean)
    mean_diff = math_utils.nanadd(other.mean, -self._mean)
    self._var = (
        prev_count_ratio * self._var
        + other_count_ratio * other.var
        + prev_count_ratio * delta_mean**2
        + other_count_ratio * mean_diff**2
    )

  def result(self) -> types.NumbersT:
    return self

  def __str__(self):
    return (
        f'count: {self.count}, total: {self.total}, mean: {self.mean}, '
        f'var: {self.var}, stddev: {self.stddev}'
    )


class Var(MeanAndVariance):

  def result(self) -> types.NumbersT:
    return self.var

  def __str__(self):
    return f'var: {self.var}'


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

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__, self.batch_score_fn, self.axis)

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
class ValueAccumulator(base.CallableMetric):
  """This stores and accumulates all the values."""

  concat_fn: Callable[[Any, Any], Any] | None = None
  metric_fns: Callable[..., Any] | dict[str, Callable[..., Any]] | None = None
  _data: tuple[list[Any], ...] = dataclasses.field(default=(), kw_only=True)

  @property
  def data(self) -> tuple[list[Any], ...]:
    return self._data

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__, self.concat_fn, self.metric_fns)

  def new(self, *args):
    if self.concat_fn:
      return self.__class__(_data=tuple(x for x in args))
    return self.__class__(_data=tuple([x] for x in args))

  def merge(self, other: Self) -> None:
    if not self._data:
      self._data = tuple(other.data)
      return
    xs_and_ys = zip(self._data, other.data, strict=True)
    if self.concat_fn:
      self._data = tuple(self.concat_fn(x, y) for x, y in xs_and_ys)
    else:
      self._data = tuple(x + y for x, y in xs_and_ys)

  def result(self) -> tuple[list[Any], ...] | list[Any] | dict[str, Any] | Any:
    if not self.metric_fns:
      return self._data if len(self._data) > 1 else self._data[0]
    if isinstance(self.metric_fns, dict):
      return {k: metric(*self._data) for k, metric in self.metric_fns.items()}
    return self.metric_fns(*self._data)


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

  Attributes:
    sum_y_true: The sum of y_true.
    sum_y_pred: The sum of y_true * y_pred.
    sum_neg_y_true: The sum of 1 - y_true.
    sum_neg_y_pred: The sum of (1 - y_true) * y_pred.
  """

  sum_y_true: float = 0
  sum_y_pred: float = 0
  sum_neg_y_true: float = 0
  sum_neg_y_pred: float = 0

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__)

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


@dataclasses.dataclass
class _PartialCrossFeatureStats(abc.ABC, base.MergeableMetric):
  """Partial cross feature statistics."""
  num_samples: int = 0
  sum_x: types.NumbersT = 0
  sum_y: float = 0
  sum_xx: types.NumbersT = 0  # sum(x**2)
  sum_yy: float = 0  # sum(y**2)
  sum_xy: types.NumbersT = 0  # sum(x * y)

  def __eq__(self, other: '_PartialCrossFeatureStats') -> bool:
    return (
        self.num_samples == other.num_samples
        and self.sum_x == other.sum_x
        and self.sum_y == other.sum_y
        and self.sum_xx == other.sum_xx
        and self.sum_yy == other.sum_yy
        and self.sum_xy == other.sum_xy
    )

  def add(
      self, x: types.NumbersT, y: types.NumbersT
  ) -> '_PartialCrossFeatureStats':
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

  def merge(
      self, other: '_PartialCrossFeatureStats'
  ) -> '_PartialCrossFeatureStats':
    self.num_samples += other.num_samples
    self.sum_x += other.sum_x
    self.sum_y += other.sum_y
    self.sum_xx += other.sum_xx
    self.sum_yy += other.sum_yy
    self.sum_xy += other.sum_xy

    return self

  @abc.abstractmethod
  def result(self) -> types.NumbersT:
    """Must be overwritten by the specific metric."""
    pass


class Covariance(_PartialCrossFeatureStats):
  """Computes the covariance of two sets of data.

  Covariance = E[(X-E[X]) * (Y-E[Y])] = E[XY] - E[X] * E[Y]
  https://en.wikipedia.org/wiki/Covariance
  """

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(
        self.__class__,
        sum_x=self.sum_x,
        sum_y=self.sum_y,
        sum_xx=self.sum_xx,
        sum_yy=self.sum_yy,
        sum_xy=self.sum_xy,
    )

  def result(self) -> types.NumbersT:
    # TODO: b/417267344 - Implement Delta Degrees of Freedom. Here, ddof is
    # always 0.

    # Covariance = E[(X-E[X]) * (Y-E[Y])] = E[XY] - E[X] * E[Y]
    # = [sum(XY) - sum(X) * sum(Y) / num_samples] / num_samples
    return (
        self.sum_xy - self.sum_x * self.sum_y / self.num_samples
    ) / self.num_samples


@dataclasses.dataclass
class RRegression(_PartialCrossFeatureStats):
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

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(
        self.__class__,
        sum_x=self.sum_x,
        sum_y=self.sum_y,
        sum_xx=self.sum_xx,
        sum_yy=self.sum_yy,
        sum_xy=self.sum_xy,
        center=self.center,
    )

  def result(self) -> types.NumbersT:
    """Calculates the Pearson Correlation Coefficient (PCC).

    PCC = cov(X, Y) / std(X) / std(Y)
    where cov is the covariance, and std(X) is the standard deviation of X;
    and analogously for std(Y).

    After substituting estimates of the covariances and variances
    PCC = sum((x_i - x_bar) * (y_i - y_bar))
          / sqrt(sum((x_i - x_bar)**2)) / sqrt(sum((x_i - x_bar)**2))
    where x_i and y_i are the individual sampled points indexed with i, and
    x_bar is the sample mean; and analogously for y_bar.

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

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(self.__class__)

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
