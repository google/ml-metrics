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
"""Cross Entropy Metrics."""

from ml_metrics._src.aggregates import types
from ml_metrics._src.utils import math_utils
import numpy as np


def binary_cross_entropy(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
    from_logits: bool = False,
    label_smoothing: float = 0,
) -> float:
  """Binary Cross Entropy Metric.

  Args:
    y_true: Truth label. This is either 0 or 1.
    y_pred: Predicted value. This is the model's prediction, i.e, a single
      floating-point value which either represents a logit, (i.e., value in
      (-inf, inf) when from_logits=True) or a probability (i.e., value in [0.,
      1.] when from_logits=False).
    from_logits: If 'y_pred' is logit values. By default, we assume that
      'y_pred' is probabilities (i.e., values in [0, 1]).
    label_smoothing: Float in range [0, 1]. When 0, no smoothing occurs. When >
      0, we compute the loss between the predicted labels and a smoothed version
      of the true labels, where the smoothing squeezes the labels towards 0.5.
      Larger values of label_smoothing correspond to heavier smoothing.

  Returns:
    The binary cross-entropy loss between true labels and predicted labels.
  """
  # Smooth labels.
  labels = y_true * (1 - label_smoothing) + 0.5 * label_smoothing

  # Cross entropy loss =
  #   -mean(y_true * log(p(y_true)) + (1 - y) * log(1 - p(y_true)))
  # where p(y) is the predicted probability that the y is 1.
  if from_logits:
    # p(y) = sigmoid(logits) = sigmoid(y_pred)
    y_pred = math_utils.sigmoid(y_pred)

  # p(y) = y_pred
  elementwise_cross_entropy = -np.multiply(
      labels, np.log(y_pred)
  ) - np.multiply((1 - labels), np.log(1 - y_pred))

  return np.mean(elementwise_cross_entropy)


def categorical_cross_entropy(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
    from_logits: bool = False,
    label_smoothing: float = 0,
):
  """Categorical Cross Entropy Metric.

  Args:
    y_true: Truth label. This is either 0 or 1.
    y_pred: Predicted value. This is the model's prediction, i.e, a single
      floating-point value which either represents a logit, (i.e., value in
      (-inf, inf) when from_logits=True) or a probability (i.e., value in [0.,
      1.] when from_logits=False).
    from_logits: If 'y_pred' is logit values. By default, we assume that
      'y_pred' is probabilities (i.e., values in [0, 1]).
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
      meaning the confidence on label values are relaxed. For example, if '0.1',
      use '0.1 / num_classes' for non-target labels and
      '0.9 + 0.1 / num_classes' for target labels.

  Returns:
    The categorical cross-entropy loss between true labels and predicted labels.
  """
  # Smooth labels.
  y_true = y_true * (1.0 - label_smoothing) + label_smoothing / y_true.shape[0]

  # Cross entropy loss = -sum(y_true * log(normalized(p(y_pred)))
  # where p(y) is the predicted probability that the y is 1.
  if from_logits:
    # p(y) = exp(logits) = exp(y_pred)
    y_pred = np.exp(y_pred)

  # p(y) = y_pred
  return -np.sum(np.multiply(y_true, np.log(np.divide(y_pred, np.sum(y_pred)))))
