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

import collections
from collections.abc import Sequence
import dataclasses
from typing import Any

import chainable
from ml_metrics._src.aggregates import classification
from ml_metrics._src.aggregates import types
from ml_metrics._src.metrics import utils
from ml_metrics.google.tools.signal_registry import registry
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np


_StrOrMetric = classification.ConfusionMatrixMetric | str

CalibrationHistogramResult = collections.namedtuple(
    'CalibrationHistogramResult',
    ('num_examples_hist', 'labels_hist', 'predictions_hist', 'bin_edges'),
)


# TODO: b/368067018 - Inherit from ml_metrics._src.aggregates.stats.Histogram.
@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    usage_category=telemetry.CATEGORY.METRIC,
)
@dataclasses.dataclass
class CalibrationHistogram(chainable.MergeableMetric):
  """Computes the Histogram of the inputs.

  Examples:
    >>> hist = CalibrationHistogram(bins=2, range=(0, 1))
    >>> hist.add([0.1], [0.2])
    >>> hist.result()

  Attributes:
    range: The lower and upper range of the bins. e.g. range = (0, 1).
    bins: The number of buckets to use.
    _num_examples_hist: The values of the histogram.
    _labels_hist: The values of the histogram for labels.
    _predictions_hist: The values of the histogram for predictions.
    _bin_edges: The bin edges of the histogram. All but the right-most bin are
      half-open. I.e. if the bins_edges are (0, 1, 2, 3, ..., 8, 9, 10), then
      the bin ranges are [0, 1), [1, 2), [2, 3), ... [8, 9), [9, 10].
  """

  range: tuple[float, float] = (0, 1)
  bins: int = 10000
  _num_examples_hist: np.ndarray = dataclasses.field(init=False)
  _labels_hist: np.ndarray = dataclasses.field(init=False)
  _predictions_hist: np.ndarray = dataclasses.field(init=False)
  _bin_edges: np.ndarray = dataclasses.field(init=False)

  def __post_init__(self):
    default_hist, self._bin_edges = np.histogram(
        a=(), bins=self.bins, range=self.range
    )

    self._num_examples_hist = self._labels_hist = self._predictions_hist = (
        default_hist
    )

  @property
  def num_examples_hist(self) -> np.ndarray:
    return self._num_examples_hist

  @property
  def labels_hist(self) -> np.ndarray:
    return self._labels_hist

  @property
  def predictions_hist(self) -> np.ndarray:
    return self._predictions_hist

  @property
  def bin_edges(self) -> np.ndarray:
    return self._bin_edges

  # TODO: b/366063413 - Replace this with a batch_weights_fn.
  def _get_histograms_and_bin_edges(
      self, labels: types.NumbersT, predictions: types.NumbersT
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_examples_hist, bin_edges = np.histogram(
        np.concatenate((labels, predictions)),
        bins=self.bins,
        range=self.range,
    )

    labels_hist, _ = np.histogram(labels, bins=bin_edges, weights=labels)
    predictions_hist, _ = np.histogram(
        predictions, bins=bin_edges, weights=predictions
    )

    return num_examples_hist, labels_hist, predictions_hist, bin_edges

  def _merge(
      self,
      num_examples_hist: np.ndarray,
      labels_hist: np.ndarray,
      predictions_hist: np.ndarray,
      bin_edges: np.ndarray,
  ) -> 'CalibrationHistogram':
    if not np.array_equal(bin_edges, self._bin_edges):
      # Self histo and new histo have different bin edges.
      raise ValueError(
          'The bin edges of the two Histograms must be equal, but recieved'
          f' self._bin_edges={self._bin_edges} and new_bin_edges={bin_edges}.'
      )

    self._num_examples_hist = self._num_examples_hist + num_examples_hist
    self._labels_hist = self._labels_hist + labels_hist
    self._predictions_hist = self._predictions_hist + predictions_hist

    return self

  def add(
      self, labels: types.NumbersT, predictions: types.NumbersT
  ) -> 'CalibrationHistogram':
    num_examples_hist, labels_hist, predictions_hist, new_bin_edges = (
        self._get_histograms_and_bin_edges(
            labels=labels, predictions=predictions
        )
    )
    return self._merge(
        num_examples_hist, labels_hist, predictions_hist, new_bin_edges
    )

  def merge(self, other: 'CalibrationHistogram') -> 'CalibrationHistogram':
    return self._merge(
        other.num_examples_hist,
        other.labels_hist,
        other.predictions_hist,
        other.bin_edges,
    )

  def result(self) -> CalibrationHistogramResult:
    return CalibrationHistogramResult(
        num_examples_hist=self._num_examples_hist.copy(),
        labels_hist=self._labels_hist.copy(),
        predictions_hist=self._predictions_hist.copy(),
        bin_edges=self._bin_edges.copy(),
    )


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    usage_category=telemetry.CATEGORY.METRIC,
)
class ClassificationAggFn(chainable.AggregateFn):
  """Wrapper over the Classification AggFn classes.

  Examples:
    >>> agg = ClassificationAggFn(metrics=['precision'])
    >>> agg.create_state()
  """

  agg_fn: chainable.AggregateFn

  def __init__(
      self,
      metrics: Sequence[_StrOrMetric] | _StrOrMetric,
      *,
      pos_label: bool | int | str | bytes = 1,
      input_type: types.InputType = types.InputType.BINARY,
      average: types.AverageType = types.AverageType.BINARY,
      vocab: dict[str, int] | None = None,
      dtype: type[Any] | None = None,
      k_list: Sequence[int] | None = None,
  ):
    """Initializes the instance.

    Args:
      metrics: List of CFM metrics.
      pos_label: The class to report if average='binary' and the data is binary.
      input_type: one input type from types.InputType.
      average: one average type from types.AverageType.
      vocab: an external vocabulary that maps categorical value to integer class
        id.
      dtype: dtype of the confusion matrix and all computations.
      k_list: k_list is only applicable for average_type != Samples and
        multiclass/multioutput input types.
    """
    if average == types.AverageType.SAMPLES:
      if k_list:
        raise ValueError('k_list is not supported for average=SAMPLES')
      self.agg_fn = classification.SamplewiseConfusionMatrixAggFn(
          vocab=vocab,
          dtype=dtype,
          metrics=metrics,
          pos_label=pos_label,
          input_type=input_type,
      )
    else:
      if k_list:
        self.agg_fn = classification.TopKConfusionMatrixAggFn(
            vocab=vocab,
            average=average,
            dtype=dtype,
            metrics=metrics,
            pos_label=pos_label,
            input_type=input_type,
            k_list=k_list,
        )
      else:
        self.agg_fn = classification.ConfusionMatrixAggFn(
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
      self,
      state: classification.ConfusionMatrixAggState | None,
      *inputs: Any,
  ):
    return self.agg_fn.update_state(state, *inputs)

  def get_result(self, state):
    return self.agg_fn.get_result(state)

  def merge_states(self, states):
    return self.agg_fn.merge_states(states)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def classification_metrics(
    metrics: Sequence[_StrOrMetric] | _StrOrMetric,
    *,
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> dict[_StrOrMetric, float]:
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
    Dict containing the evaluation metric values.

  Examples:
    >>> classification_metrics(['precision'], y_true=[0, 1], y_pred=[0, 1])
    {'precision': 1.0}
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


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def precision(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Precision classification metric.

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

  Examples:
    >>> precision([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.PRECISION,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def ppv(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute PPV classification metric.

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

  Examples:
    >>> ppv([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.PPV,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def recall(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Recall classification metric.

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

  Examples:
    >>> recall([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.RECALL,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def f1_score(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute F1 Score classification metric.

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

  Examples:
    >>> f1_score([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.F1_SCORE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def accuracy(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Accuracy classification metric.

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

  Examples:
    >>> accuracy([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.ACCURACY,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def binary_accuracy(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Binary Accuracy classification metric.

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

  Examples:
    >>> binary_accuracy([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.BINARY_ACCURACY,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def sensitivity(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Sensitivity classification metric.

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

  Examples:
    >>> sensitivity([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.SENSITIVITY,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def tpr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute TPR (True Positive rate/sensitivity) classification metric.

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

  Examples:
    >>> tpr([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.TPR,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def specificity(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Specificity classification metric.

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

  Examples:
    >>> specificity([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.SPECIFICITY,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def tnr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute TNR (True negative rate) classification metric.

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

  Examples:
    >>> tnr([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.TNR,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def fall_out(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Fall-out classification metric.

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

  Examples:
    >>> fall_out([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.FALL_OUT,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def fpr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute FPR (False Positive rate) classification metric.

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

  Examples:
    >>> fpr([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.FPR,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def miss_rate(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Miss Rate classification metric.

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

  Examples:
    >>> miss_rate([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.MISS_RATE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def fnr(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute FNR (False Negative Rate) classification metric.

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

  Examples:
    >>> fnr([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.FNR,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def negative_predictive_value(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Negative Predictive Value classification metric.

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

  Examples:
    >>> negative_predictive_value([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.NEGATIVE_PREDICTIVE_VALUE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def npv(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute alias of Negative Predictive Value classification metric.

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

  Examples:
    >>> npv([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.NPV,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def false_discovery_rate(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute False Discovery Rate classification metric.

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

  Examples:
    >>> false_discovery_rate([0, 1], [0, 1])
    (0.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.FALSE_DISCOVERY_RATE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def false_omission_rate(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute False Omission Rate classification metric.

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

  Examples:
    >>> false_omission_rate([0, 1], [0, 1])
    (0.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.FALSE_OMISSION_RATE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def threat_score(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Threat Score classification metric.

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

  Examples:
    >>> threat_score([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.THREAT_SCORE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def positive_likelihood_ratio(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Positive Likelihood Ratio classification metric.

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

  Examples:
    >>> positive_likelihood_ratio([0, 1], [0, 1])
    (inf,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.POSITIVE_LIKELIHOOD_RATIO,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def negative_likelihood_ratio(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Negative Likelihood Ratio classification metric.

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

  Examples:
    >>> negative_likelihood_ratio([0, 1], [0, 1])
    (0.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.NEGATIVE_LIKELIHOOD_RATIO,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def diagnostic_odds_ratio(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Diagnostic Odds Ratio classification metric.

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

  Examples:
    >>> diagnostic_odds_ratio([0, 1], [0, 1])
    (inf,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.DIAGNOSTIC_ODDS_RATIO,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def positive_predictive_value(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Positive Predictive Value classification metric.

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

  Examples:
    >>> positive_predictive_value([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.POSITIVE_PREDICTIVE_VALUE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def intersection_over_union(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Intersection over Union classification metric.

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

  Examples:
    >>> intersection_over_union([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.INTERSECTION_OVER_UNION,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def prevalence(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Prevalence classification metric.

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

  Examples:
    >>> prevalence([0, 1], [0, 1])
    (0.5,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.PREVALENCE,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def prevalence_threshold(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Prevalence Threshold classification metric.

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

  Examples:
    >>> prevalence_threshold([0, 1], [0, 1])
    (0.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.PREVALENCE_THRESHOLD,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def matthews_correlation_coefficient(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Matthews Correlation Coefficient classification metric.

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

  Examples:
    >>> matthews_correlation_coefficient([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.MATTHEWS_CORRELATION_COEFFICIENT,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def informedness(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Informedness classification metric.

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

  Examples:
    >>> informedness([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.INFORMEDNESS,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def markedness(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Markedness classification metric.

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

  Examples:
    >>> markedness([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.MARKEDNESS,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def balanced_accuracy(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: types.InputType = types.InputType.BINARY,
    average: types.AverageType = types.AverageType.BINARY,
    vocab: dict[str, int] | None = None,
    dtype: type[Any] | None = None,
    k_list: Sequence[int] | None = None,
) -> tuple[float, ...]:
  """Compute Balanced Accuracy classification metric.

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

  Examples:
    >>> balanced_accuracy([0, 1], [0, 1])
    (1.0,)
  """
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return ClassificationAggFn(
      metrics=classification.ConfusionMatrixMetric.BALANCED_ACCURACY,
      pos_label=pos_label,
      input_type=input_type,
      average=average,
      vocab=vocab,
      dtype=dtype,
      k_list=k_list,
  )(y_true, y_pred)
