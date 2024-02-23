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
"""Individual Classification based metrics."""

from collections.abc import Sequence
from typing import Any

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import classification
from ml_metrics._src.aggregates import types
from ml_metrics._src.metrics import utils

InputType = types.InputType
AverageType = types.AverageType
ConfusionMatrixMetric = classification.ConfusionMatrixMetric
ConfusionMatrixAggFn = classification.ConfusionMatrixAggFn
TopKConfusionMatrixAggFn = classification.TopKConfusionMatrixAggFn
SamplewiseConfusionMatrixAggFn = classification.SamplewiseConfusionMatrixAggFn

_METRIC_PYDOC_POSTFIX = """

  Args:
    y_true: array of sample's true labels
    y_pred: array of sample's label predictions
    pos_label: The class to report if average='binary' and the data is binary.
      By default it is 1. Please set in case this default is not a valid label.
      If the data are multiclass or multilabel, this will be ignored.
    input_type: one input type from types.InputType
    average: one average  type from types.AverageType
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    dtype: dtype of the confusion matrix and all computations. Default to None
      as it is inferred.
    k_list: k_list is only applicable for average_type != Samples and
      multiclass/multioutput input types. It is a list of topk each of which
      slices y_pred by y_pred[:topk] assuming the predictions are sorted in
      descending order. Default 'None' means consider all outputs in the
      prediction.

  Returns:
    Tuple with metric value(s)
"""


class ClassificationAggFn(base.AggregateFn):
  """Wrapper over the Classification AggFn classes."""

  agg_fn: base.AggregateFn

  def __init__(
      self,
      metrics: Sequence[ConfusionMatrixMetric],
      *,
      pos_label: bool | int | str | bytes = 1,
      input_type: InputType = InputType.BINARY,
      average: AverageType = AverageType.BINARY,
      vocab: dict[str, int] | None = None,
      dtype: type[Any] | None = None,
      k_list: Sequence[int] | None = None,
  ):
    if average == AverageType.SAMPLES:
      if k_list:
        raise ValueError('k_list is not supported for average=SAMPLES')
      self.agg_fn = SamplewiseConfusionMatrixAggFn(
          vocab=vocab,
          dtype=dtype,
          metrics=metrics,
          pos_label=pos_label,
          input_type=input_type,
      )
    else:
      if k_list:
        self.agg_fn = TopKConfusionMatrixAggFn(
            vocab=vocab,
            average=average,
            dtype=dtype,
            metrics=metrics,
            pos_label=pos_label,
            input_type=input_type,
            k_list=k_list,
        )
      else:
        self.agg_fn = ConfusionMatrixAggFn(
            vocab=vocab,
            average=average,
            dtype=dtype,
            metrics=metrics,
            pos_label=pos_label,
            input_type=input_type,
        )

  def create_state(self) -> Any:
    return self.agg_fn.create_state()

  def update_state(
      self, state: classification.ConfusionMatrixAggState | None, *inputs: Any
  ):
    return self.agg_fn.update_state(state, *inputs)

  def get_result(self, state):
    return self.agg_fn.get_result(state)

  def merge_states(self, states):
    return self.agg_fn.merge_states(states)


def compute_metrics(
    metrics: Sequence[ConfusionMatrixMetric],
    *,
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[tuple[float, ...], ...]:
  """Compute multiple metrics together for better efficiency.

  Args:
    metrics: List of CFM metrics
    y_true: array of sample's true labels
    y_pred: array of sample's label predictions
    pos_label: The class to report if average='binary' and the data is binary.
      By default it is 1. Please set in case this default is not a valid label.
      If the data are multiclass or multilabel, this will be ignored.
    input_type: one input type from types.InputType
    average: one average  type from types.AverageType
    vocab: an external vocabulary that maps categorical value to integer class
      id. This is required if computed distributed (when merge_accumulators is
      called) and the average is macro where the class id mapping needs to be
      stable.
    dtype: dtype of the confusion matrix and all computations. Default to None
      as it is inferred.
    k_list: k_list is only applicable for average_type != Samples and
      multiclass/multioutput input types. It is a list of topk each of which
      slices y_pred by y_pred[:topk] assuming the predictions are sorted in
      descending order. Default 'None' means consider all outputs in the
      prediction.

  Returns:
    Tuple containing the evaluation metric values. in the corresponding order of
      given metric names in metrics list.
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


def precision(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Precision classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.PRECISION,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
precision.__doc__ += _METRIC_PYDOC_POSTFIX


def ppv(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute PPV classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.PPV,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
ppv.__doc__ += _METRIC_PYDOC_POSTFIX


def recall(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Recall classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.RECALL,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
recall.__doc__ += _METRIC_PYDOC_POSTFIX


def f1_score(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute F1 Score classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.F1_SCORE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
f1_score.__doc__ += _METRIC_PYDOC_POSTFIX


def accuracy(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Accuracy classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.ACCURACY,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
accuracy.__doc__ += _METRIC_PYDOC_POSTFIX


def binary_accuracy(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Binary Accuracy classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.BINARY_ACCURACY,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
binary_accuracy.__doc__ += _METRIC_PYDOC_POSTFIX


def sensitivity(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Sensitivity classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.SENSITIVITY,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
sensitivity.__doc__ += _METRIC_PYDOC_POSTFIX


def tpr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute TPR (True Positive rate/sensitivity) classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.TPR,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
tpr.__doc__ += _METRIC_PYDOC_POSTFIX


def specificity(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Specificity classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.SPECIFICITY,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
specificity.__doc__ += _METRIC_PYDOC_POSTFIX


def tnr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute TNR (True negative rate) classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.TNR,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
tnr.__doc__ += _METRIC_PYDOC_POSTFIX


def fall_out(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Fall-out classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.FALL_OUT,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
fall_out.__doc__ += _METRIC_PYDOC_POSTFIX


def fpr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute FPR (False Positive rate) classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.FPR,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
fpr.__doc__ += _METRIC_PYDOC_POSTFIX


def miss_rate(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Miss Rate classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.MISS_RATE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
miss_rate.__doc__ += _METRIC_PYDOC_POSTFIX


def fnr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute FNR (False Negative Rate) classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.FNR,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
fnr.__doc__ += _METRIC_PYDOC_POSTFIX


def negative_prediction_value(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Negative Prediction Value classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.NEGATIVE_PREDICTION_VALUE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
negative_prediction_value.__doc__ += _METRIC_PYDOC_POSTFIX


def nvp(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute alias of Negative Prediction Value classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.NVP,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
nvp.__doc__ += _METRIC_PYDOC_POSTFIX


def false_discovery_rate(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute False Discovery Rate classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.FALSE_DISCOVERY_RATE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
false_discovery_rate.__doc__ += _METRIC_PYDOC_POSTFIX


def false_omission_rate(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute False Omission Rate classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.FALSE_OMISSION_RATE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
false_omission_rate.__doc__ += _METRIC_PYDOC_POSTFIX


def threat_score(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Threat Score classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.THREAT_SCORE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
threat_score.__doc__ += _METRIC_PYDOC_POSTFIX


def positive_likelihood_ratio(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Positive Likelihood Ratio classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.POSITIVE_LIKELIHOOD_RATIO,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
positive_likelihood_ratio.__doc__ += _METRIC_PYDOC_POSTFIX


def negative_likelihood_ratio(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Negative Likelihood Ratio classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.NEGATIVE_LIKELIHOOD_RATIO,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
negative_likelihood_ratio.__doc__ += _METRIC_PYDOC_POSTFIX


def diagnostic_odds_ratio(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Diagnostic Odds Ratio classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.DIAGNOSTIC_ODDS_RATIO,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
diagnostic_odds_ratio.__doc__ += _METRIC_PYDOC_POSTFIX


def positive_predictive_value(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Positive Predictive Value classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.POSITIVE_PREDICTIVE_VALUE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
positive_predictive_value.__doc__ += _METRIC_PYDOC_POSTFIX


def intersection_over_union(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Intersection over Union classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.INTERSECTION_OVER_UNION,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
intersection_over_union.__doc__ += _METRIC_PYDOC_POSTFIX


def prevalence(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Prevalence classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.PREVALENCE,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
prevalence.__doc__ += _METRIC_PYDOC_POSTFIX


def prevalence_threshold(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Prevalence Threshold classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.PREVALENCE_THRESHOLD,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
prevalence_threshold.__doc__ += _METRIC_PYDOC_POSTFIX


def matthews_correlation_coefficient(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Matthews Correlation Coefficient classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.MATTHEWS_CORRELATION_COEFFICIENT,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
matthews_correlation_coefficient.__doc__ += _METRIC_PYDOC_POSTFIX


def informedness(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Informedness classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.INFORMEDNESS,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
informedness.__doc__ += _METRIC_PYDOC_POSTFIX


def markedness(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Markedness classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.MARKEDNESS,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
markedness.__doc__ += _METRIC_PYDOC_POSTFIX


def balanced_accuracy(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Balanced Accuracy classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=(ConfusionMatrixMetric.BALANCED_ACCURACY,),
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)[0]
balanced_accuracy.__doc__ += _METRIC_PYDOC_POSTFIX
