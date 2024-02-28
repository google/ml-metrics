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

import collections
from collections.abc import Sequence
import dataclasses

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import types
from ml_metrics._src.aggregates import utils
import numpy as np

InputType = types.InputType

MeanState = utils.MeanState
MeanStatesPerMetric = dict[str, MeanState]


class RetrievalMetric(types.StrEnum):  # pylint: disable=invalid-enum-extension
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


safe_divide = utils.safe_divide


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
  return safe_divide(2 * precision * recall, (precision + recall))


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
  return utils.pos_sqrt(precision * recall)


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


@dataclasses.dataclass(kw_only=True)
class TopKRetrievalConfig(base.MetricMaker):
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


@dataclasses.dataclass(kw_only=True)
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

  def add(self, y_trues, y_preds):
    """Compute all true positive related metrics."""
    k_list = list(sorted(self.k_list)) if self.k_list else [float('inf')]
    y_pred_count = np.asarray([len(row) for row in y_preds])
    y_true_count = np.asarray([len(row) for row in y_trues])
    max_pred_count = max(y_pred_count)
    max_pred_count = min(max_pred_count, max(k_list))
    tp = []
    for y_pred, y_true in zip(y_preds, y_trues):
      tp.append([
          int(y_pred[i] in y_true) if i < len(y_pred) else 0
          for i in range(max_pred_count)
      ])
    tp = np.asarray(tp)
    # True positives at TopK is of a dimension of Examples x K as the following:
    # The first dimension is always batch dimension (# examples), the second
    # dimension can be either 0D (single-output) or 1D (multioutput) array.
    # E.g., provided one multioutput prediction and its true positive:
    # topk:         [1,    2,     3]
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
      self._state['accuracy'] += MeanState(
          accuracy.sum(axis=0), accuracy.shape[0]
      )
      result['accuracy'] = accuracy

    precision, recall = None, None
    if 'precision' in self.metrics:
      precision = _precision(tp_at_topks, k_list, y_pred_count)
      self._state['precision'] += MeanState(
          precision.sum(axis=0), precision.shape[0]
      )
      result['precision'] = precision

    if 'ppv' in self.metrics:
      ppv = _ppv(tp_at_topks, k_list, y_pred_count)
      self._state['ppv'] += MeanState(ppv.sum(axis=0), ppv.shape[0])
      result['ppv'] = ppv

    if 'recall' in self.metrics:
      recall = _recall(tp_at_topks, k_list, y_true_count)
      self._state['recall'] += MeanState(recall.sum(axis=0), recall.shape[0])
      result['recall'] = recall

    if 'sensitivity' in self.metrics:
      sensitivity = _sensitivity(tp_at_topks, k_list, y_true_count)
      self._state['sensitivity'] += MeanState(
          sensitivity.sum(axis=0), sensitivity.shape[0]
      )
      result['sensitivity'] = sensitivity

    if 'tpr' in self.metrics:
      tpr = _tpr(tp_at_topks, k_list, y_true_count)
      self._state['tpr'] += MeanState(tpr.sum(axis=0), tpr.shape[0])
      result['tpr'] = tpr

    if 'positive_predictive_value' in self.metrics:
      positive_predictive_value = _positive_predictive_value(
          tp_at_topks, k_list, y_pred_count
      )
      self._state['positive_predictive_value'] += MeanState(
          positive_predictive_value.sum(axis=0),
          positive_predictive_value.shape[0],
      )
      result['positive_predictive_value'] = positive_predictive_value

    if 'intersection_over_union' in self.metrics:
      intersection_over_union = _intersection_over_union(
          tp_at_topks, k_list, y_true_count, y_pred_count
      )
      self._state['intersection_over_union'] += MeanState(
          intersection_over_union.sum(axis=0), intersection_over_union.shape[0]
      )
      result['intersection_over_union'] = intersection_over_union

    if 'f1_score' in self.metrics:
      if precision is None:
        precision = _precision(tp_at_topks, k_list, y_pred_count)
      if recall is None:
        recall = _recall(tp_at_topks, k_list, y_true_count)
      f1 = _f1_score(precision, recall)
      self._state['f1_score'] += MeanState(f1.sum(axis=0), f1.shape[0])
      result['f1_score'] = f1

    if 'mean_average_precision' in self.metrics:
      mean_average_precision = _mean_average_precision(
          tp, tp_at_topks, k_range, k_list, y_true_count
      )
      self._state['mean_average_precision'] += MeanState(
          mean_average_precision.sum(axis=0),
          mean_average_precision.shape[0],
      )
      result['mean_average_precision'] = mean_average_precision

    if 'mean_reciprocal_rank' in self.metrics:
      reciprocal_ranks = _mean_reciprocal_rank(tp_at_topks, k_list)
      self._state['mean_reciprocal_rank'] += MeanState(
          reciprocal_ranks.sum(axis=0), reciprocal_ranks.shape[0]
      )
      result['reciprocal_ranks'] = reciprocal_ranks

    if 'miss_rate' in self.metrics:
      miss_rate = _miss_rate(tp_at_topks, k_list, y_true_count)
      self._state['miss_rate'] += MeanState(
          miss_rate.sum(axis=0), miss_rate.shape[0]
      )
      result['miss_rate'] = miss_rate

    if 'false_discovery_rate' in self.metrics:
      false_discovery_rate = _false_discovery_rate(
          tp_at_topks, k_list, y_pred_count
      )
      self._state['false_discovery_rate'] += MeanState(
          false_discovery_rate.sum(axis=0), false_discovery_rate.shape[0]
      )
      result['false_discovery_rate'] = false_discovery_rate

    if 'threat_score' in self.metrics:
      threat_score = _threat_score(tp_at_topks, k_list, y_true_count)
      self._state['threat_score'] += MeanState(
          threat_score.sum(axis=0), threat_score.shape[0]
      )
      result['threat_score'] = threat_score

    if 'fowlkes_mallows_index' in self.metrics:
      fowlkes_mallows_index = _fowlkes_mallows_index(
          tp_at_topks, k_list, y_true_count, y_pred_count
      )
      self._state['fowlkes_mallows_index'] += MeanState(
          fowlkes_mallows_index.sum(axis=0), fowlkes_mallows_index.shape[0]
      )
      result['fowlkes_mallows_index'] = fowlkes_mallows_index

    if 'dcg_score' in self.metrics:
      dcg = _dcg_score(tp, k_range, k_list)
      self._state['dcg_score'] += MeanState(dcg.sum(axis=0), dcg.shape[0])
      result['dcg_score'] = dcg

    if 'ndcg_score' in self.metrics:
      ndcg = _ndcg_score(tp, k_range, k_list, y_true_count)
      self._state['ndcg_score'] += MeanState(ndcg.sum(axis=0), ndcg.shape[0])
      result['ndcg_score'] = ndcg
    return result

  def merge(self, other: 'TopKRetrieval'):
    for metric in self.metrics:
      self._state[metric] += other.state[metric]

  def result(self):
    result = [self._state[metric].result() for metric in self.metrics]
    # Extends the remaining Ks from the last value.
    if self.k_list and result and len(self.k_list) > len(result[0]):
      for i, metric_result in enumerate(result):
        extra_ks = len(self.k_list) - len(metric_result)
        result[i] = list(metric_result) + [metric_result[-1]] * extra_ks
    return result


def TopKRetrievalAggFn(**kwargs):  # pylint: disable=invalid-name
  """Convenient alias as a AggregateFn constructor."""
  return base.MergibleMetricAggFn(metric_maker=TopKRetrievalConfig(**kwargs))
