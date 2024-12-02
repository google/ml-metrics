# Copyright 2023 Google LLC
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

"""Aggregates modules for all retrieval metrics."""
from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import dataclasses
import enum
import functools
import itertools
from typing import Any

from ml_metrics._src import base_types
from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import types
from ml_metrics._src.aggregates import utils
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import math_utils
import numpy as np


InputType = types.InputType

MeanState = utils.MeanState
MeanStatesPerMetric = dict[str, MeanState]


class RetrievalMetric(enum.StrEnum):  # pylint: disable=invalid-enum-extension
  """Supported retrieval metrics."""

  CONFUSION_MATRIX = 'confusion_matrix'
  PRECISION = 'precision'
  PPV = 'ppv'
  RECALL = 'recall'
  SENSITIVITY = 'sensitivity'
  TPR = 'tpr'
  ACCURACY = 'accuracy'
  POSITIVE_PREDICTIVE_VALUE = 'positive_predictive_value'
  INTERSECTION_OVER_UNION = 'intersection_over_union'
  F1_SCORE = 'f1_score'
  MISS_RATE = 'miss_rate'
  FALSE_DISCOVERY_RATE = 'false_discovery_rate'
  THREAT_SCORE = 'threat_score'
  FOWLKES_MALLOWS_INDEX = 'fowlkes_mallows_index'
  MEAN_AVERAGE_PRECISION = 'mean_average_precision'
  MEAN_RECIPROCAL_RANK = 'mean_reciprocal_rank'
  DCG_SCORE = 'dcg_score'  # Discounted Cumulative Gain
  NDCG_SCORE = 'ndcg_score'  # Normalized Discounted Cumulative Gain


def _accuracy(tp_at_topks, k_list):
  return (tp_at_topks[:, k_list - 1] > 0).astype(np.int32)


def _precision(tp_at_topks, k_list, y_pred_count):
  return tp_at_topks[:, k_list - 1] / np.minimum(
      k_list, y_pred_count[:, np.newaxis]
  )


def _ppv(tp_at_topks, k_list, y_pred_count):
  """Alias for Precision."""
  return _precision(tp_at_topks, k_list, y_pred_count)


def _recall(tp_at_topks, k_list, y_true_len):
  return tp_at_topks[:, k_list - 1] / y_true_len[:, np.newaxis]


def _sensitivity(tp_at_topks, k_list, y_true_len):
  """Sensitivity."""
  return _recall(tp_at_topks, k_list, y_true_len)


def _tpr(tp_at_topks, k_list, y_true_len):
  """Alias for Sensitivity."""
  return _recall(tp_at_topks, k_list, y_true_len)


def _positive_predictive_value(tp_at_topks, k_list, y_pred_count):
  return tp_at_topks[:, k_list - 1] / np.minimum(
      k_list, y_pred_count[:, np.newaxis]
  )


def _intersection_over_union(tp_at_topks, k_list, y_true_len, y_pred_count):
  return tp_at_topks[:, k_list - 1] / (
      np.minimum(k_list, y_pred_count[:, np.newaxis])
      + y_true_len[:, np.newaxis]
      - tp_at_topks[:, k_list - 1]
  )


def _f1_score(precision, recall):
  return math_utils.safe_divide(2 * precision * recall, (precision + recall))


def _miss_rate(tp_at_topks, k_list, y_true_len):
  return 1 - _recall(tp_at_topks, k_list, y_true_len)


def _false_discovery_rate(tp_at_topks, k_list, y_pred_count):
  return 1 - _precision(tp_at_topks, k_list, y_pred_count)


def _threat_score(tp_at_topks, k_list, y_true_len):
  cumsum_fn = y_true_len[:, np.newaxis] - tp_at_topks[:, k_list - 1]
  return tp_at_topks[:, k_list - 1] / (cumsum_fn + k_list)


def _fowlkes_mallows_index(tp_at_topks, k_list, y_true_len, y_pred_count):
  precision = _precision(tp_at_topks, k_list, y_pred_count)
  recall = _recall(tp_at_topks, k_list, y_true_len)
  return math_utils.pos_sqrt(precision * recall)


def _mean_average_precision(tp, tp_at_topks, ks, k_list, y_true_len):
  precision_all_k = tp_at_topks[:, ks - 1] / ks
  # Average Precision = (precision[k] * relevance[k]) / K where we
  # use tp > 0 as a proxy for relevance.
  relevance = tp > 0
  size_true = np.minimum(ks, y_true_len[:, np.newaxis])
  result = np.cumsum(precision_all_k * relevance, axis=1) / size_true
  result = result[:, k_list - 1]
  return result


def _mean_reciprocal_rank(tp_at_topks, k_list):
  # The index of the first non-zero true positive.
  ranks = np.argmax(tp_at_topks > 0, axis=1) + 1
  # Assign infinity to the false positives as their ranks.
  ranks = np.where(tp_at_topks > 0, ranks[:, np.newaxis], np.inf)
  result = (1.0 / ranks)[:, k_list - 1]
  return result


def _dcg_score(tp, k_range, k_list):
  """Discounted Cumulative Gain."""
  # Hard coded the relevance to 1.0.
  discounted_gain = 1.0 / np.log2(k_range + 1)
  discounted_cumulative_gain = np.cumsum(
      np.where(tp > 0, discounted_gain, 0.0), axis=1
  )
  return discounted_cumulative_gain[:, k_list - 1]


def _ndcg_score(tp, k_range, k_list, y_true_count):
  """Normalized Discounted Cumulative Gain."""
  # Hard coded the relevance to 1.0.
  discounted_gain = 1.0 / np.log2(k_range + 1)
  discounted_cumulative_gain = np.cumsum(
      np.where(tp > 0, discounted_gain, 0.0), axis=1
  )
  ideal_discounted_gain = np.where(
      k_range > y_true_count[:, np.newaxis], 0.0, discounted_gain
  )
  ideal_discounted_cumulative_gain = np.cumsum(ideal_discounted_gain, axis=1)
  result = (
      discounted_cumulative_gain[:, k_list - 1]
      / ideal_discounted_cumulative_gain[:, k_list - 1]
  )
  return result


class RetrievalMetricAtThreshold(str):
  """Retrieval metric at threshold in a format of metric@threshold.

  This will parse a metric from a string to two parts of metric and threshold.
  The metric is one of the supported retrieval metrics. The threshold is a float
  number. If the threshold is not provided, it will be set to None.
  """

  metric: RetrievalMetric | None
  threshold: float | None

  def __init__(self, metric_name: str):
    splitted = metric_name.lower().strip().split('@')
    if len(splitted) == 2:
      self.metric = RetrievalMetric(splitted[0])
      self.threshold = float(splitted[1])
    elif len(splitted) == 1:
      self.metric = RetrievalMetric(splitted[0])
      self.threshold = None
    else:
      raise ValueError(f'Invalid metric: {metric_name}')


def retrieval_matcher(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
    y_prob: types.NumbersT | None = None,
) -> tuple[types.NumbersT, types.NumbersT, types.NumbersT]:
  """Matches a batched y_true and y_pred optionally with y_prob.

  This outputs the matched in the shape of y_true and y_pred with the matching
  entry set to the corresponding probability from y_prob if provided or 1 when
  y_prob is not provided.

  Args:
    y_true: The ground truth.
    y_pred: The prediction.
    y_prob: The probability of the prediction.

  Returns:
    matched_true_prob: The matched probability of the ground truth.
    matched_pred_prob: The matched probability of the prediction.
    matched_y_prob: Same as y_pro but autofilled with 1s when not provided.
  """
  matched_true_prob, matched_pred_prob, matched_y_prob = [], [], []
  y_prob = y_prob or [None] * len(y_true)
  for row_true, row_pred, row_prob in zip(y_true, y_pred, y_prob, strict=True):
    row_prob = (
        np.ones_like(row_pred, dtype=np.float32)
        if row_prob is None
        else np.asarray(row_prob)
    )
    row_true, row_pred = np.asarray(row_true), np.asarray(row_pred)
    row_true_prob = np.zeros_like(row_true, dtype=np.float32)
    row_pred_prob = np.zeros_like(row_pred, dtype=np.float32)
    for i, (prob, pred) in enumerate(zip(row_prob, row_pred)):
      ixs = np.where(row_true == pred)[0]
      assert len(ixs) < 2
      if ixs.size > 0:
        row_pred_prob[i] = prob
        row_true_prob[ixs[0]] = prob
    matched_true_prob.append(row_true_prob)
    matched_pred_prob.append(row_pred_prob)
    matched_y_prob.append(row_prob)
  return matched_true_prob, matched_pred_prob, matched_y_prob


@dataclasses.dataclass(kw_only=True)
class _ThresholdedConfusionMatrix:
  """RetrievalMetricConfig."""

  thresholds: types.NumbersT = 0
  tp_trues: types.NumbersT = 0
  tp_preds: types.NumbersT = 0
  p_trues: types.NumbersT = 0
  p_preds: types.NumbersT = 0

  def merge(self, other):
    assert all(self.thresholds == other.thresholds)
    self.tp_trues += other.tp_trues
    self.tp_preds += other.tp_preds
    self.p_trues += other.p_trues
    self.p_preds += other.p_preds

  @functools.cached_property
  def precision(self):
    return math_utils.safe_divide(self.tp_preds, self.p_preds)

  @functools.cached_property
  def recall(self):
    return math_utils.safe_divide(self.tp_trues, self.p_trues)

  @functools.cached_property
  def f1_score(self):
    return _f1_score(self.precision, self.recall)

  def get_metric(self, metric: RetrievalMetricAtThreshold):
    if metric.threshold is None:
      return getattr(self, metric.metric)
    else:
      return np.interp(
          metric.threshold, self.thresholds, getattr(self, metric.metric)
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ThresholdedRetrieval(base.MergeableMetric):
  """TopKRetrievals with continuous input.

  A typical use case is to calculate the retrieval metrics but with thresholds.
  y_true = [[id1, id2, id4], ...]
  y_pred = [[id1, id3, id2, id4], ...]
  y_prob = [[0.9, 0.8, 0.7, 0.2], ...]
  The matcher should give me a filtered y_prob to both sides:
  y_true_prob = [[0.9, 0.7, 0.2], ...].
  y_pred_prob = [[0.9, 0.0, 0.7, 0.2], ...], note that mismatched id is set to
  0.
  If we only are interested in id1:
  y_true_prob = [[0.9, 0.0, 0.0]].
  y_pred_prob = [[0.9, 0.0, 0.0, 0.0]],
  Given a threshold = 0.75:
  y_true_mask = [[True]]], sum(y_true_mask) is true_p_count
  y_pred_mask = [[True]], sum(y_pred_mask) is pred_p_count

  This is a binary multi-output retrieval problem.
  The input is tp_pred, tp_true, and y_prob and thresholds.
  When the slicing is different, e.g., the distances are different, we need to
  tp_pred, but how about tp_true, yes, it will also be masked.  Then what does
  each threshold mean for tp_true? When tp_pred is masked to False, the
  corresponding one in y_true should also be masked to False.

  Attributes:
    thresholds: The thresholds to be used for the retrieval metrics.
    metrics: The metrics to be computed.
    state: The mergeable metric state.
  """

  thresholds: types.NumbersT | Sequence[float] = (0.0,)
  metrics: Sequence[RetrievalMetric | RetrievalMetricAtThreshold] = (
      RetrievalMetric.PRECISION,
      RetrievalMetric.RECALL,
      RetrievalMetric.F1_SCORE,
  )
  matcher: Callable[..., Any] | None = retrieval_matcher
  _confusion_matrix: _ThresholdedConfusionMatrix = dataclasses.field(
      default_factory=_ThresholdedConfusionMatrix, init=False
  )

  def __post_init__(self):
    thresholds = np.asarray(sorted(self.thresholds), dtype=np.float32)
    object.__setattr__(self, 'thresholds', thresholds)
    confusion_matrix = _ThresholdedConfusionMatrix(thresholds=thresholds)
    object.__setattr__(self, '_confusion_matrix', confusion_matrix)
    metrics = [RetrievalMetricAtThreshold(metric) for metric in self.metrics]
    object.__setattr__(self, 'metrics', metrics)

  @property
  def confusion_matrix(self):
    return self._confusion_matrix

  def add(
      self,
      y_true=None,
      y_pred=None,
      y_prob=None,
      matched_true_prob=None,
      matched_pred_prob=None,
  ) -> _ThresholdedConfusionMatrix:
    """Compute all true positive counts from the y_true and y_pred.

    If y_true_prob and y_pred_prob are provided, this skips the internal matcher
    logic. This is useful when mathcer is expensive and is called externally to
    avoid repeated computations.

    Args:
      y_true: The ground truth.
      y_pred: The prediction.
      y_prob: The probability of the prediction.
      matched_true_prob: The matched probability of the ground truth, all
        negative values are filtered out.
      matched_pred_prob: The matched probability of the prediction, all negative
        values are filtered out.

    Returns:
      ThresholdedConfusionMatrix for this batch.
    """
    if matched_true_prob is None or matched_pred_prob is None:
      matched_true_prob, matched_pred_prob, y_prob = self.matcher(
          y_true, y_pred, y_prob
      )
    # Flatten the 2D list of list to 1D array.
    y_prob = np.array(list(itertools.chain(*y_prob)))
    matched_true_prob = np.array(list(itertools.chain(*matched_true_prob)))
    matched_pred_prob = np.array(list(itertools.chain(*matched_pred_prob)))
    matched_true_prob = matched_true_prob[matched_true_prob >= 0]
    matched_pred_prob = matched_pred_prob[matched_pred_prob >= 0]
    # 2D array of true positives at each threshold with a dimension of
    # num_thresholds x num_y.
    y_trues = np.array(
        [matched_true_prob > threshold for threshold in self.thresholds]
    )
    y_preds = np.array(
        [matched_pred_prob > threshold for threshold in self.thresholds]
    )
    y_probs = np.array([y_prob > threshold for threshold in self.thresholds])
    tp_trues, true_p_count = y_trues.sum(axis=1), len(matched_true_prob)
    tp_preds, pred_p_count = y_preds.sum(axis=1), y_probs.sum(axis=1)
    confusion_matrix = _ThresholdedConfusionMatrix(
        thresholds=self.thresholds,
        tp_trues=tp_trues,
        tp_preds=tp_preds,
        p_trues=true_p_count,
        p_preds=pred_p_count,
    )
    self._confusion_matrix.merge(confusion_matrix)
    return confusion_matrix

  def merge(self, other: ThresholdedRetrieval):
    self._confusion_matrix.merge(other.confusion_matrix)

  def result(self):
    """Returns the metrics."""
    result = {'thresholds': self.thresholds}
    assert self._confusion_matrix is not None
    for metric in self.metrics:
      result[metric] = self._confusion_matrix.get_metric(metric)
    return result


@dataclasses.dataclass(kw_only=True, frozen=True)
class TopKRetrievalConfig(base_types.Makeable):
  """TopKRetrievals.

  Attributes:
    k_list: topk list, default to None, which means all outputs are considered.
    metrics: The metrics to be computed.
    input_type: input encoding type, must be `multiclass` or
      `multiclass-multioutput`.
  """

  k_list: Sequence[int] | None = None
  metrics: Sequence[RetrievalMetric] = (
      RetrievalMetric.PRECISION,
      RetrievalMetric.PPV,
      RetrievalMetric.RECALL,
      RetrievalMetric.SENSITIVITY,
      RetrievalMetric.TPR,
      RetrievalMetric.POSITIVE_PREDICTIVE_VALUE,
      RetrievalMetric.INTERSECTION_OVER_UNION,
      RetrievalMetric.F1_SCORE,
      RetrievalMetric.ACCURACY,
      RetrievalMetric.MEAN_AVERAGE_PRECISION,
      RetrievalMetric.MEAN_RECIPROCAL_RANK,
      RetrievalMetric.MISS_RATE,
      RetrievalMetric.FALSE_DISCOVERY_RATE,
      RetrievalMetric.THREAT_SCORE,
      RetrievalMetric.FOWLKES_MALLOWS_INDEX,
      RetrievalMetric.DCG_SCORE,
      RetrievalMetric.NDCG_SCORE,
  )
  input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT

  def __post_init__(self):
    if self.input_type not in (
        InputType.MULTICLASS_MULTIOUTPUT,
        InputType.MULTICLASS,
    ):
      raise NotImplementedError(f'"{str(self.input_type)}" is not supported.')

  def make(self):
    return TopKRetrieval(k_list=self.k_list, metrics=self.metrics)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TopKRetrieval(base.MergeableMetric):
  """TopKRetrievals.

  Attributes:
    k_list: topk list, default to None, which means all outputs are considered.
    metrics: The metrics to be computed.
  """

  k_list: Sequence[int] | None = None
  metrics: Sequence[RetrievalMetric] = ()
  _state: MeanStatesPerMetric = dataclasses.field(
      default_factory=lambda: collections.defaultdict(MeanState),
      init=False,
  )

  @property
  def state(self):
    return self._state

  def add(self, y_true, y_pred) -> dict[str, types.NumbersT]:
    """Compute all true positive related metrics."""
    k_list = list(sorted(self.k_list)) if self.k_list else [float('inf')]
    y_pred_count = np.asarray([len(row) for row in y_pred])
    y_true_count = np.asarray([len(row) for row in y_true])
    max_pred_count = max(y_pred_count)
    max_pred_count = min(max_pred_count, max(k_list))
    tp = []
    for y_pred_row, y_true_row in zip(y_pred, y_true):
      tp.append([
          int(y_pred_row[i] in y_true_row) if i < len(y_pred_row) else 0
          for i in range(max_pred_count)
      ])
    tp = np.asarray(tp)
    # True positives at TopK is of a dimension of Examples x K as the following:
    # The first dimension is always batch dimension (# examples), the second
    # dimension can be either 0D (single-output) or 1D (multioutput) array.
    # E.g., provided one multioutput prediction and its true positive:
    # topk:          [1,    2,     3]
    # true-positive: [True, False, True]
    # We will get the true positives at topKs:
    # topk:     [top1, top2, top3]
    # tp_topks: [1,     1,      2]
    tp_at_topks = np.cumsum(tp, axis=1)
    # Truncates the k_list with maximum length of the predictions.
    k_list = np.asarray(
        [k for k in k_list if k < max_pred_count] + [max_pred_count]
    )

    # A consecutive K list that is useful to calculate average-over-Ks metrics
    # such as mean average precision.
    k_range = np.arange(max_pred_count) + 1

    result = {}
    if 'accuracy' in self.metrics:
      accuracy = _accuracy(tp_at_topks, k_list)
      self._state['accuracy'].merge(
          MeanState(accuracy.sum(axis=0), accuracy.shape[0])
      )
      result['accuracy'] = accuracy

    precision, recall = None, None
    if 'precision' in self.metrics:
      precision = _precision(tp_at_topks, k_list, y_pred_count)
      self._state['precision'].merge(
          MeanState(precision.sum(axis=0), precision.shape[0])
      )
      result['precision'] = precision

    if 'ppv' in self.metrics:
      ppv = _ppv(tp_at_topks, k_list, y_pred_count)
      self._state['ppv'].merge(MeanState(ppv.sum(axis=0), ppv.shape[0]))
      result['ppv'] = ppv

    if 'recall' in self.metrics:
      recall = _recall(tp_at_topks, k_list, y_true_count)
      self._state['recall'].merge(
          MeanState(recall.sum(axis=0), recall.shape[0])
      )
      result['recall'] = recall

    if 'sensitivity' in self.metrics:
      sensitivity = _sensitivity(tp_at_topks, k_list, y_true_count)
      self._state['sensitivity'].merge(
          MeanState(sensitivity.sum(axis=0), sensitivity.shape[0])
      )
      result['sensitivity'] = sensitivity

    if 'tpr' in self.metrics:
      tpr = _tpr(tp_at_topks, k_list, y_true_count)
      self._state['tpr'].merge(MeanState(tpr.sum(axis=0), tpr.shape[0]))
      result['tpr'] = tpr

    if 'positive_predictive_value' in self.metrics:
      positive_predictive_value = _positive_predictive_value(
          tp_at_topks, k_list, y_pred_count
      )
      self._state['positive_predictive_value'].merge(
          MeanState(
              positive_predictive_value.sum(axis=0),
              positive_predictive_value.shape[0],
          )
      )
      result['positive_predictive_value'] = positive_predictive_value

    if 'intersection_over_union' in self.metrics:
      intersection_over_union = _intersection_over_union(
          tp_at_topks, k_list, y_true_count, y_pred_count
      )
      self._state['intersection_over_union'].merge(
          MeanState(
              intersection_over_union.sum(axis=0),
              intersection_over_union.shape[0]
          )
      )
      result['intersection_over_union'] = intersection_over_union

    if 'f1_score' in self.metrics:
      if precision is None:
        precision = _precision(tp_at_topks, k_list, y_pred_count)
      if recall is None:
        recall = _recall(tp_at_topks, k_list, y_true_count)
      f1 = _f1_score(precision, recall)
      self._state['f1_score'].merge(MeanState(f1.sum(axis=0), f1.shape[0]))
      result['f1_score'] = f1

    if 'mean_average_precision' in self.metrics:
      mean_average_precision = _mean_average_precision(
          tp, tp_at_topks, k_range, k_list, y_true_count
      )
      self._state['mean_average_precision'].merge(
          MeanState(
              mean_average_precision.sum(axis=0),
              mean_average_precision.shape[0],
          )
      )
      result['mean_average_precision'] = mean_average_precision

    if 'mean_reciprocal_rank' in self.metrics:
      reciprocal_ranks = _mean_reciprocal_rank(tp_at_topks, k_list)
      self._state['mean_reciprocal_rank'].merge(
          MeanState(
              reciprocal_ranks.sum(axis=0),
              reciprocal_ranks.shape[0]
          )
      )
      result['reciprocal_ranks'] = reciprocal_ranks

    if 'miss_rate' in self.metrics:
      miss_rate = _miss_rate(tp_at_topks, k_list, y_true_count)
      self._state['miss_rate'].merge(
          MeanState(miss_rate.sum(axis=0), miss_rate.shape[0])
      )
      result['miss_rate'] = miss_rate

    if 'false_discovery_rate' in self.metrics:
      false_discovery_rate = _false_discovery_rate(
          tp_at_topks, k_list, y_pred_count
      )
      self._state['false_discovery_rate'].merge(
          MeanState(
              false_discovery_rate.sum(axis=0),
              false_discovery_rate.shape[0]
          )
      )
      result['false_discovery_rate'] = false_discovery_rate

    if 'threat_score' in self.metrics:
      threat_score = _threat_score(tp_at_topks, k_list, y_true_count)
      self._state['threat_score'].merge(
          MeanState(threat_score.sum(axis=0), threat_score.shape[0])
      )
      result['threat_score'] = threat_score

    if 'fowlkes_mallows_index' in self.metrics:
      fowlkes_mallows_index = _fowlkes_mallows_index(
          tp_at_topks, k_list, y_true_count, y_pred_count
      )
      self._state['fowlkes_mallows_index'].merge(
          MeanState(
              fowlkes_mallows_index.sum(axis=0),
              fowlkes_mallows_index.shape[0]
          )
      )
      result['fowlkes_mallows_index'] = fowlkes_mallows_index

    if 'dcg_score' in self.metrics:
      dcg = _dcg_score(tp, k_range, k_list)
      self._state['dcg_score'].merge(MeanState(dcg.sum(axis=0), dcg.shape[0]))
      result['dcg_score'] = dcg

    if 'ndcg_score' in self.metrics:
      ndcg = _ndcg_score(tp, k_range, k_list, y_true_count)
      self._state['ndcg_score'].merge(
          MeanState(ndcg.sum(axis=0), ndcg.shape[0])
      )
      result['ndcg_score'] = ndcg
    return result

  def merge(self, other: 'TopKRetrieval'):
    for metric in self.metrics:
      self._state[metric].merge(other.state[metric])

  def result(self):
    result = [self._state[metric].result() for metric in self.metrics]
    # Extends the remaining Ks from the last value.
    if self.k_list and result and len(self.k_list) > len(result[0]):
      for i, metric_result in enumerate(result):
        extra_ks = len(self.k_list) - len(metric_result)
        result[i] = list(metric_result) + [metric_result[-1]] * extra_ks
    return tuple(result)


def TopKRetrievalAggFn(**kwargs):  # pylint: disable=invalid-name
  """Convenient alias as a AggregateFn constructor."""
  return base.MergeableMetricAggFn(TopKRetrievalConfig(**kwargs))

lazy_fns.makeables.register(TopKRetrievalConfig, base.MergeableMetricAggFn)
