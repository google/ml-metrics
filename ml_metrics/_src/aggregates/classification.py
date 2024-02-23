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
"""Aggregates modules for all classification metrics."""

from collections.abc import Iterable, Sequence
import dataclasses
import itertools
from typing import Any

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import types
from ml_metrics._src.aggregates import utils
import numpy as np

AverageType = types.AverageType
InputType = types.InputType


class ConfusionMatrixMetric(types.StrEnum):  # pylint: disable=invalid-enum-extension
  """Confusion Matrix Metrics that can be composed from CM's intermediate state."""
  CONFUSION_MATRIX = 'confusion_matrix'
  PRECISION = 'precision'
  PPV = 'ppv'
  RECALL = 'recall'
  F1_SCORE = 'f1_score'
  ACCURACY = 'accuracy'
  BINARY_ACCURACY = 'binary_accuracy'
  SENSITIVITY = 'sensitivity'
  TPR = 'tpr'
  SPECIFICITY = 'specificity'
  TNR = 'tnr'
  FALL_OUT = 'fall_out'
  FPR = 'fpr'
  MISS_RATE = 'miss_rate'
  FNR = 'fnr'
  NEGATIVE_PREDICTION_VALUE = 'negative_prediction_value'
  NVP = 'nvp'
  FALSE_DISCOVERY_RATE = 'false_discovery_rate'
  FALSE_OMISSION_RATE = 'false_omission_rate'
  THREAT_SCORE = 'threat_score'
  POSITIVE_LIKELIHOOD_RATIO = 'positive_likelihood_ratio'
  NEGATIVE_LIKELIHOOD_RATIO = 'negative_likelihood_ratio'
  DIAGNOSTIC_ODDS_RATIO = 'diagnostic_odds_ratio'
  POSITIVE_PREDICTIVE_VALUE = 'positive_predictive_value'
  INTERSECTION_OVER_UNION = 'intersection_over_union'
  PREVALENCE = 'prevalence'
  PREVALENCE_THRESHOLD = 'prevalence_threshold'
  MATTHEWS_CORRELATION_COEFFICIENT = 'matthews_correlation_coefficient'
  INFORMEDNESS = 'informedness'
  MARKEDNESS = 'markedness'
  BALANCED_ACCURACY = 'balanced_accuracy'
  MEAN_AVERAGE_PRECISION = 'mean_average_precision'


class _ConfusionMatrix:
  """Confusion Matrix accumulator with kwargs used to compute it.

  Confusion matrix for classification and retrieval tasks.
  See https://en.wikipedia.org/wiki/Confusion_matrix for more info.

  Attributes:
    tp: True positives count.
    tn: True negatives count.
    fp: False positives count.
    fn: False negative count.
    average: The average type of which this confusion matrix is computed.
    dtype: data type of the instance numpy array. None by default, dtype is
      deduced by numpy at construction.
  """

  def __init__(
      self,
      tp: types.NumbersT,
      tn: types.NumbersT,
      fp: types.NumbersT,
      fn: types.NumbersT,
      *,
      dtype: type[Any] | None = None,
  ):
    self.tp = np.asarray(tp, dtype=dtype)
    self.tn = np.asarray(tn, dtype=dtype)
    self.fp = np.asarray(fp, dtype=dtype)
    self.fn = np.asarray(fn, dtype=dtype)
    self.dtype = dtype

  @property
  def t(self):
    """Labeled True count."""
    return self.tp + self.fn

  @property
  def p(self):
    """Predicted True count."""
    return self.tp + self.fp

  def __iadd__(self, other):
    self.tp += other.tp
    self.tn += other.tn
    self.fp += other.fp
    self.fn += other.fn
    return self

  def __add__(self, other):
    tp = self.tp + other.tp
    tn = self.tn + other.tn
    fp = self.fp + other.fp
    fn = self.fn + other.fn
    return _ConfusionMatrix(tp, tn, fp, fn)

  def __eq__(self, other):
    """Numerically equals."""
    return (
        np.allclose(self.tp, other.tp)
        and np.allclose(self.tn, other.tn)
        and np.allclose(self.fp, other.fp)
        and np.allclose(self.fn, other.fn)
    )

  def __repr__(self):
    return f'tp={self.tp}, tn={self.tn}, fp={self.fp}, fn={self.fn}'

  def derive_metric(
      self, metric: ConfusionMatrixMetric, average=None
  ) -> types.NumbersT:
    """Helper to call the right metric function given a Metric Enum."""
    match metric:
      case ConfusionMatrixMetric.PRECISION:
        result = _precision(self)
      case ConfusionMatrixMetric.PPV:
        result = _ppv(self)
      case ConfusionMatrixMetric.RECALL:
        result = _recall(self)
      case ConfusionMatrixMetric.F1_SCORE:
        result = _f1(self)
      case ConfusionMatrixMetric.ACCURACY:
        result = _accuracy(self)
      case ConfusionMatrixMetric.BINARY_ACCURACY:
        result = _binary_accuracy(self)
      case ConfusionMatrixMetric.SENSITIVITY:
        result = _sensitivity(self)
      case ConfusionMatrixMetric.TPR:
        result = _tpr(self)
      case ConfusionMatrixMetric.SPECIFICITY:
        result = _specificity(self)
      case ConfusionMatrixMetric.TNR:
        result = _tnr(self)
      case ConfusionMatrixMetric.FALL_OUT:
        result = _fall_out(self)
      case ConfusionMatrixMetric.FPR:
        result = _fpr(self)
      case ConfusionMatrixMetric.MISS_RATE:
        result = _miss_rate(self)
      case ConfusionMatrixMetric.FNR:
        result = _fnr(self)
      case ConfusionMatrixMetric.NEGATIVE_PREDICTION_VALUE:
        result = _negative_prediction_value(self)
      case ConfusionMatrixMetric.NVP:
        result = _npv(self)
      case ConfusionMatrixMetric.FALSE_DISCOVERY_RATE:
        result = _false_discovery_rate(self)
      case ConfusionMatrixMetric.FALSE_OMISSION_RATE:
        result = _false_omission_rate(self)
      case ConfusionMatrixMetric.THREAT_SCORE:
        result = _threat_score(self)
      case ConfusionMatrixMetric.POSITIVE_LIKELIHOOD_RATIO:
        result = _positive_likelihood_ratio(self)
      case ConfusionMatrixMetric.NEGATIVE_LIKELIHOOD_RATIO:
        result = _negative_likelihood_ratio(self)
      case ConfusionMatrixMetric.DIAGNOSTIC_ODDS_RATIO:
        result = _diagnostic_odds_ratio(self)
      case ConfusionMatrixMetric.POSITIVE_PREDICTIVE_VALUE:
        result = _positive_predictive_value(self)
      case ConfusionMatrixMetric.INTERSECTION_OVER_UNION:
        result = _intersection_over_union(self)
      case ConfusionMatrixMetric.PREVALENCE:
        result = _prevalence(self)
      case ConfusionMatrixMetric.PREVALENCE_THRESHOLD:
        result = _prevalence_threshold(self)
      case ConfusionMatrixMetric.MATTHEWS_CORRELATION_COEFFICIENT:
        result = _matthews_correlation_coefficient(self)
      case ConfusionMatrixMetric.INFORMEDNESS:
        result = _informedness(self)
      case ConfusionMatrixMetric.MARKEDNESS:
        result = _markedness(self)
      case ConfusionMatrixMetric.BALANCED_ACCURACY:
        result = _balanced_accuracy(self)
      case _:
        raise NotImplementedError(f'"{metric}" metric is not supported.')
    assert (
        average != AverageType.SAMPLES
    ), 'Unexpected samplewise average for a derived metric.'
    if average is None or average in ('micro', 'binary'):
      return result
    elif average == 'macro':
      return np.mean(result, axis=0)
    else:
      raise NotImplementedError(f'"{average}" average is not supported.')


class _TopKConfusionMatrix(_ConfusionMatrix):
  """Confusion Matrix accumulator with kwargs used to compute it.

  Confusion matrix for classification and retrieval tasks.
  See https://en.wikipedia.org/wiki/Confusion_matrix for more info.

  Attributes:
    k: a sequence of topk, sequentially corresponds to the actual confusion
      matrix counts (tp, tn, fp, fn).
    tp: True positives count.
    tn: True negatives count.
    fp: False positives count.
    fn: False negative count.
    dtype: data type of the instance numpy array. None by default, dtype is
      deduced by numpy at construction.
  """

  def __init__(
      self,
      k: types.NumbersT,
      tp: types.NumbersT,
      tn: types.NumbersT,
      fp: types.NumbersT,
      fn: types.NumbersT,
      *,
      dtype: type[Any] | None = None,
  ):
    self.k = np.asarray(k, dtype=dtype)
    super().__init__(tp, tn, fp, fn, dtype=dtype)

  def __eq__(self, other):
    """Numerically equals."""
    return np.allclose(self.k, other.k) and super().__eq__(other)

  def __str__(self):
    return f'k={self.k}, tp={self.tp}, tn={self.tn}, fp={self.fp}, fn={self.fn}'


def _precision(cm: _ConfusionMatrix):
  return utils.safe_divide(cm.tp, cm.p)


def _ppv(cm: _ConfusionMatrix):
  """Alias for Precision."""
  return _precision(cm)


def _recall(cm: _ConfusionMatrix):
  return utils.safe_divide(cm.tp, cm.t)


def _f1(cm: _ConfusionMatrix):
  precision = _precision(cm)
  recall = _recall(cm)
  return utils.safe_divide(2 * precision * recall, precision + recall)


def _accuracy(cm: _ConfusionMatrix) -> types.NumbersT:
  """Accuracy, only meaningful for samplewise ConfusionMatrixAtK."""
  return (cm.tp > 0).astype(int)


def _binary_accuracy(cm: _ConfusionMatrix) -> types.NumbersT:
  """Binary accuracy."""
  return utils.safe_divide(cm.tp + cm.tn, cm.tp + cm.fp + cm.tn + cm.fn)


def _sensitivity(cm: _ConfusionMatrix) -> types.NumbersT:
  """Sensitivity."""
  return _recall(cm)


def _tpr(cm: _ConfusionMatrix) -> types.NumbersT:
  """True positive rate."""
  return _recall(cm)


def _specificity(cm: _ConfusionMatrix) -> types.NumbersT:
  """Specificity or Selectivity."""
  return utils.safe_divide(cm.tn, (cm.tn + cm.fp))


def _tnr(cm: _ConfusionMatrix) -> types.NumbersT:
  """True Negative rate."""
  return _specificity(cm)


def _fall_out(cm: _ConfusionMatrix) -> types.NumbersT:
  """Fall out rate."""
  return utils.safe_divide(cm.fp, (cm.fp + cm.tn))


def _fpr(cm: _ConfusionMatrix) -> types.NumbersT:
  """False positive rate."""
  return _fall_out(cm)


def _miss_rate(cm: _ConfusionMatrix) -> types.NumbersT:
  """MissRate."""
  return utils.safe_divide(cm.fn, (cm.fn + cm.tp))


def _fnr(cm: _ConfusionMatrix) -> types.NumbersT:
  """Alias for MissRate."""
  return _miss_rate(cm)


def _negative_prediction_value(cm: _ConfusionMatrix) -> types.NumbersT:
  """Negative predictive value (NPV)."""
  return utils.safe_divide(cm.tn, (cm.tn + cm.fn))


def _npv(cm: _ConfusionMatrix) -> types.NumbersT:
  """Alias for Negative predictive value."""
  return _negative_prediction_value(cm)


def _false_discovery_rate(cm: _ConfusionMatrix) -> types.NumbersT:
  """False discovery rate (FDR)."""
  return utils.safe_divide(cm.fp, (cm.p))


def _false_omission_rate(cm: _ConfusionMatrix) -> types.NumbersT:
  """False discovery rate (FDR)."""
  return utils.safe_divide(cm.fn, (cm.fn + cm.tn))


def _threat_score(cm: _ConfusionMatrix) -> types.NumbersT:
  """Threat score or critical success index (TS or CSI)."""
  return utils.safe_divide(cm.tp, (cm.t + cm.fp))


def _positive_likelihood_ratio(cm: _ConfusionMatrix) -> types.NumbersT:
  """Postive Likelihood ratio."""
  return utils.safe_divide(_tpr(cm), _fpr(cm))


def _negative_likelihood_ratio(cm: _ConfusionMatrix) -> types.NumbersT:
  """Negative Likelihodd ratio."""
  return utils.safe_divide(_fnr(cm), _tnr(cm))


def _diagnostic_odds_ratio(cm: _ConfusionMatrix) -> types.NumbersT:
  """Diagnostic Odds ratio (tp*tn/fp*fn)."""
  return utils.safe_divide(
      _positive_likelihood_ratio(cm), _negative_likelihood_ratio(cm)
  )


def _positive_predictive_value(cm: _ConfusionMatrix) -> types.NumbersT:
  return utils.safe_divide(cm.tp, cm.p)


def _intersection_over_union(cm: _ConfusionMatrix) -> types.NumbersT:
  return utils.safe_divide(cm.tp, (cm.t + cm.fp))


def _prevalence(cm: _ConfusionMatrix) -> types.NumbersT:
  """Prevalence."""
  return utils.safe_divide((cm.tp + cm.fn), (cm.tp + cm.tn + cm.fn + cm.fp))


def _prevalence_threshold(cm: _ConfusionMatrix) -> types.NumbersT:
  """Prevalence threshold (PT)."""
  tnr = _tnr(cm)
  tpr = _tpr(cm)
  return utils.safe_divide(
      (utils.pos_sqrt(tpr * (1 - tnr)) + tnr - 1), (tpr + tnr - 1)
  )


def _matthews_correlation_coefficient(cm: _ConfusionMatrix) -> types.NumbersT:
  """Matthews corrrelation coefficient (MCC)."""
  numerator = cm.tp * cm.tn - cm.fp * cm.fn
  denominator = utils.pos_sqrt(
      (cm.tp + cm.fp) * (cm.tp + cm.fn) * (cm.tn + cm.fp) * (cm.tn + cm.fn)
  )
  return utils.safe_divide(numerator, denominator)


def _informedness(cm: _ConfusionMatrix) -> types.NumbersT:
  """Informedness or bookmaker informedness (BM)."""
  return _tpr(cm) + _tnr(cm) - 1


def _markedness(cm: _ConfusionMatrix) -> types.NumbersT:
  """Markedness (MK) or deltaP."""
  return _ppv(cm) + _npv(cm) - 1


def _balanced_accuracy(cm: _ConfusionMatrix) -> types.NumbersT:
  return (_tpr(cm) + _tnr(cm)) / 2


def _indicator_confusion_matrix(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
    *,
    pos_label: str | int | bytes = 1,
    multiclass: bool = False,
    average: AverageType = AverageType.MICRO,
) -> _ConfusionMatrix:
  """Calculates confusion matix.

  Args:
    y_true: ground truth classification labels. This has to be encoded in one of
      the `input_type`.
    y_pred: classification preditions. This has to be encoded in one of the
      `input_type`.
    pos_label: label to be recognized as positive class, only relevant when
      input type is "binary".
    multiclass: If True, input is encoded as 2D array-like of "multi-hot"
      encoding in a shape of Batch X NumClass.
    average: average type of the confusion matrix.

  Returns:
    Confusion matrix.
  """
  if average in (AverageType.MICRO, AverageType.BINARY):
    axis = None
  elif average is None or average == AverageType.MACRO:
    axis = 0
  elif average == AverageType.SAMPLES:
    axis = 1
  else:
    raise NotImplementedError(f'"{average}" average is not supported.')

  y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
  if multiclass and (y_true.ndim != 2 or y_pred.ndim != 2):
    raise ValueError(
        'Multiclass indicator input needs to be 2D array, actual dimensions:'
        f' y_true: {y_true.ndim}; y_pred: {y_pred.ndim}'
    )

  true = y_true == pos_label
  positive = y_pred == pos_label
  # Forces the input to binary if average type is binary.
  if multiclass and average == AverageType.BINARY:
    if true.shape[1] > 2:
      raise ValueError(
          'Non-binary multiclass indicator input is not supported for `binary`'
          f' average. #classes is {true.shape[1]}'
      )
    true = true[:, 0]
    positive = positive[:, 0]
  # Reshapes to a multi-class format to Batch X Classes
  elif not multiclass and average != AverageType.BINARY:
    true = np.vstack((true, ~true)).T
    positive = np.vstack((positive, ~positive)).T
  negative = ~positive
  tp = positive & true
  fn = negative & true

  positive_cnt = positive.sum(axis=axis)
  tp_cnt = tp.sum(axis=axis)
  fp_cnt = positive_cnt - tp_cnt
  fn_cnt = fn.sum(axis=axis)
  negative_cnt = negative.sum(axis=axis)
  tn_cnt = negative_cnt - fn_cnt
  return _ConfusionMatrix(tp_cnt, tn_cnt, fp_cnt, fn_cnt)


def get_vocab(rows: Iterable[Any], multioutput: bool) -> dict[str, int]:
  """Constructs a vocabulary that maps hashables to an integer."""
  if multioutput:
    return {
        k: i for i, k in enumerate(set(itertools.chain.from_iterable(rows)))
    }
  else:
    return {k: i for i, k in enumerate(set(rows))}


def _apply_vocab(rows: Sequence[Any], vocab: dict[str, int], multioutput: bool):
  """Given a vocabulary, converts a multioutput input to a indicator output."""
  result = np.empty((len(rows), len(vocab)), dtype=np.bool_)
  result.fill(False)
  if multioutput:
    for i, row in enumerate(rows):
      for elem in row:
        result[i][vocab[elem]] = True
  else:
    for i, elem in enumerate(rows):
      result[i][vocab[elem]] = True
  return result


def _multiclass_confusion_matrix(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
    *,
    vocab: dict[str, int] | None = None,
    multioutput: bool = False,
    average: AverageType = AverageType.MICRO,
) -> _ConfusionMatrix:
  """Calculates a confusion matrix for multiclass(-multioutput) input.

  Args:
    y_true: ground truth classification labels. This has to be encoded in one of
      the `multiclass` or `multiclass-output`.
    y_pred: classification preditions. This has to be encoded in one of the
      `multiclass` or `multiclass-output`.
    vocab: an external vocabulary that maps categorical value to integer class
      id. If not provided, one is deduced within this input.
    multioutput: encoding types of the y_true and y_pred, if True, the input is
      a nested list of class identifiers.
    average: average type of the confusion matrix.

  Returns:
    Confusion matrices with k in k_list.
  """
  vocab = vocab or get_vocab(itertools.chain(y_true, y_pred), multioutput)
  y_true_dense = _apply_vocab(y_true, vocab, multioutput)
  y_pred_dense = _apply_vocab(y_pred, vocab, multioutput)
  return _indicator_confusion_matrix(
      y_true_dense,
      y_pred_dense,
      average=average,
      multiclass=True,
      pos_label=True,
  )


ConfusionMatrixAggState = _ConfusionMatrix


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfusionMatrixAggFn(base.AggregateFn):
  """ConfusionMatrix aggregate.

  Attributes:
    pos_label: the value considered as positive, default to 1.
    input_type: input encoding type, must be one of `InputType`.
    average: average type, must be one of `AverageType`.
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    dtype: dtype of the confusion matrix and all computations. Default to None
      as it is inferred.
  """

  pos_label: bool | int | str | bytes = 1
  input_type: InputType = InputType.BINARY
  # TODO(b/311208939): implements average = None.
  average: AverageType = AverageType.BINARY
  vocab: dict[str, int] | None = None
  metrics: Sequence[ConfusionMatrixMetric] = ()
  dtype: type[Any] | None = None

  def __post_init__(self):
    if self.average == AverageType.SAMPLES:
      raise ValueError(
          '"samples" average is unsupported, use the Samplewise version.'
      )

  def _calculate_confusion_matrix(
      self,
      y_true: types.NumbersT,
      y_pred: types.NumbersT,
  ) -> _ConfusionMatrix:
    if self.input_type in (InputType.MULTICLASS_INDICATOR, InputType.BINARY):
      return _indicator_confusion_matrix(
          y_true,
          y_pred,
          pos_label=self.pos_label,
          multiclass=(self.input_type == InputType.MULTICLASS_INDICATOR),
          average=self.average,
      )
    elif self.input_type in (
        InputType.MULTICLASS,
        InputType.MULTICLASS_MULTIOUTPUT,
    ):
      return _multiclass_confusion_matrix(
          y_true,
          y_pred,
          vocab=self.vocab,
          multioutput=(self.input_type == InputType.MULTICLASS_MULTIOUTPUT),
          average=self.average,
      )
    else:
      raise NotImplementedError(f'"{self.input_type}" input is not supported.')

  def update_state(
      self, state: ConfusionMatrixAggState | None, *inputs: Any
  ) -> ConfusionMatrixAggState:
    cm = self._calculate_confusion_matrix(*inputs)
    return (cm + state) if state else cm

  def merge_states(
      self, states: list[ConfusionMatrixAggState]
  ) -> ConfusionMatrixAggState:
    if (
        self.average in (AverageType.WEIGHTED, AverageType.MACRO)
        and self.vocab is None
    ):
      raise ValueError(f'Global vocab is needed for "{self.average}" average.')
    iter_acc = iter(states)
    result = next(iter_acc)
    for accumulator in iter_acc:
      result += accumulator
    return result

  def get_result(self, state: ConfusionMatrixAggState) -> Any:
    if self.metrics:
      return tuple(
          state.derive_metric(metric, average=self.average)
          for metric in self.metrics
      )
    return (state,)


def _apply_vocab_at_k(
    rows: types.NumbersT,
    vocab: dict[str, int],
    multioutput: bool,
    k_list: Sequence[int],
):
  """Encodes a multiclass(-multioutput) input in multiclass-indicator format."""
  result = np.full((len(rows), len(vocab)), False, dtype=np.bool_)
  k_list = set(k_list)
  for j in range(max(k_list)):
    if multioutput:
      for i, row in enumerate(rows):
        if j < len(row):
          result[i][vocab[row[j]]] = True
    else:
      if j == 0:
        for i, elem in enumerate(rows):
          result[i][vocab[elem]] = True
    if j + 1 in k_list:
      yield j + 1, result


def _topk_confusion_matrix(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
    *,
    k_list: Sequence[int],
    multioutput: bool,
    average: AverageType,
    vocab: dict[str, int] | None,
) -> _TopKConfusionMatrix:
  """Calculates a confusion matrix for multiclass(-multioutput) input.

  Args:
    y_true: ground truth classification labels. This has to be encoded in one of
      the `multiclass` or `multiclass-output`.
    y_pred: classification predictions. This has to be encoded in one of the
      `multiclass` or `multiclass-output`.
    k_list: a list of topk each of which slices y_pred by y_pred[:topk] assuming
      the predictions are sorted in descending order.
    multioutput: encoding types of the y_true and y_pred, if True, the input is
      a nested list of class identifiers.
    average: average type of the confusion matrix.
    vocab: an external vocabulary, if not provided, one is deduced within this
      input.

  Returns:
    Confusion matrices with k in k_list.
  """
  vocab = vocab or get_vocab(itertools.chain(y_true, y_pred), multioutput)
  y_true_dense = _apply_vocab(y_true, vocab, multioutput)
  cms = []
  for k, y_pred_dense in _apply_vocab_at_k(y_pred, vocab, multioutput, k_list):
    cm = _indicator_confusion_matrix(
        y_true_dense,
        y_pred_dense,
        average=average,
        multiclass=True,
        pos_label=True,
    )
    cms.append((k, cm.tp, cm.tn, cm.fp, cm.fn))
  return _TopKConfusionMatrix(*tuple(zip(*cms)))


@dataclasses.dataclass(kw_only=True, frozen=True)
class TopKConfusionMatrixAggFn(ConfusionMatrixAggFn):
  """ConfusionMatrixAtK aggregate.

  Attributes:
    pos_label: the value considered as positive, default to 1.
    input_type: input encoding type, must be either  `multiclass` or
      `multiclass-multioutput`.
    average: average type, must be one of the types under `AverageType`.
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    k_list: a list of topk each of which slices y_pred by y_pred[:topk] assuming
      the predictions are sorted in descending order.
  """

  k_list: Sequence[int] = ()

  def __post_init__(self):
    if self.input_type not in (
        InputType.MULTICLASS,
        InputType.MULTICLASS_MULTIOUTPUT,
    ):
      raise ValueError(
          f'"{self.input_type}" input is not supported for TopK Confusion'
          ' Matrix.'
      )

  def _calculate_confusion_matrix(
      self, y_true: types.NumbersT, y_pred: types.NumbersT
  ) -> _TopKConfusionMatrix:
    result = _topk_confusion_matrix(
        y_true,
        y_pred,
        k_list=self.k_list,
        vocab=self.vocab,
        multioutput=(self.input_type == InputType.MULTICLASS_MULTIOUTPUT),
        average=AverageType(self.average),
    )
    return result


SamplewiseConfusionMatrixAggState = dict[str, utils.MeanState]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplewiseConfusionMatrixAggFn(base.AggregateFn):
  """Samplewise ConfusionMatrix aggregate function.

  Attributes:
    pos_label: the value considered as positive, default to 1.
    input_type: input encoding type, must be one of `InputType`.
    average: fixed as `samples` average.
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    dtype: dtype of the confusion matrix and all computations. Default to None
      as it is inferred.
  """

  metrics: Sequence[ConfusionMatrixMetric]
  pos_label: bool | int | str | bytes = 1
  input_type: InputType = InputType.BINARY
  average: AverageType = dataclasses.field(
      default=AverageType.SAMPLES, init=False
  )
  vocab: dict[str, int] | None = None
  dtype: type[Any] | None = None

  def __post_init__(self):
    if self.input_type == InputType.BINARY:
      raise ValueError(
          'Samples average is not available for Binary classification.'
      )
    if self.average != AverageType.SAMPLES:
      raise ValueError(
          'Samplewise Confusion Matrix aggreation only accepts Samples Average'
          f' type, got {self.average} from {self}.'
      )

  def create_state(self) -> SamplewiseConfusionMatrixAggState:
    """Creates the initial empty state."""
    return {}

  def _metric_states(self, cm: _ConfusionMatrix) -> dict[str, utils.MeanState]:
    result = {}
    for metric in self.metrics:
      if (score := cm.derive_metric(metric)) is not None:
        result[metric] = utils.MeanState(np.sum(score), len(score))
    return result

  def _calculate_confusion_matrix(
      self,
      y_true: types.NumbersT,
      y_pred: types.NumbersT,
  ) -> _ConfusionMatrix:
    if self.input_type == InputType.MULTICLASS_INDICATOR:
      return _indicator_confusion_matrix(
          y_true,
          y_pred,
          pos_label=self.pos_label,
          multiclass=True,
          average=self.average,
      )
    elif self.input_type in (
        InputType.MULTICLASS,
        InputType.MULTICLASS_MULTIOUTPUT,
    ):
      return _multiclass_confusion_matrix(
          y_true,
          y_pred,
          vocab=self.vocab,
          multioutput=(self.input_type == InputType.MULTICLASS_MULTIOUTPUT),
          average=self.average,
      )
    else:
      raise NotImplementedError(f'"{self.input_type}" input is not supported.')

  def update_state(
      self, state: SamplewiseConfusionMatrixAggState, *inputs: Any
  ) -> SamplewiseConfusionMatrixAggState:
    """Batch updates the states of the aggregate."""
    cm = self._calculate_confusion_matrix(*inputs)
    metric_states = self._metric_states(cm)
    return self.merge_states((metric_states, state))

  def merge_states(
      self, states: Sequence[SamplewiseConfusionMatrixAggState]
  ) -> SamplewiseConfusionMatrixAggState:
    states_iter = iter(states)
    result = next(states_iter)
    for state in states_iter:
      for key, value in state.items():
        result[key] += value
    return result

  def get_result(self, state: SamplewiseConfusionMatrixAggState) -> Any:
    """Extracts the outputs from the aggregate states."""
    return tuple(state[metric].result() for metric in self.metrics)
