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
"""Individual Keras based metrics."""

from ml_metrics._src.aggregates import keras_metric_wrapper
from ml_metrics._src.aggregates import types
from ml_metrics._src.metrics import utils
import tensorflow as tf


InputType = types.InputType
AverageType = types.AverageType


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
  Returns:
    Metric value(s)
"""


def roc_auc(
    y_true,
    y_pred,
    pos_label: bool | int | str | bytes = 1,
    input_type: InputType = InputType.BINARY,
    average: AverageType = AverageType.BINARY,
    vocab: dict[str, int] | None = None
) -> tuple[float, ...]:
  """Compute Precision classification metric."""
  utils.verify_input(y_true, y_pred, average, input_type, vocab, pos_label)
  return keras_metric_wrapper.KerasAggregateFn(tf.keras.metrics.AUC())(
      y_true, y_pred
  )


roc_auc.__doc__ += _METRIC_PYDOC_POSTFIX
