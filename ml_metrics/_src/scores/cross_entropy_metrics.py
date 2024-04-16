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
import numpy as np


def binary_cross_entropy(
    y_true: types.NumbersT,
    y_pred: np.ndarray,
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

  # Cross entropy loss
  # = -mean(labels * ln(p(labels)) + (1 - labels) * ln(1 - p(labels)))
  # where p(y) is the predicted probability that the y is 1.

  # If from_logits: p(y) = sigmoid(y_true)
  # else: p(y) = y_pred

  if from_logits:
    # p(labels) = sigmoid(y_pred)

    # elementwise_cross_entropy
    # = labels * -ln(sigmoid(y_pred)) + (1 - labels) * -ln(1 - sigmoid(y_pred))
    # = labels * -ln(1 / (1 + exp(-y_pred)))
    #   + (1 - labels) * -ln(exp(-y_pred) / (1 + exp(-y_pred)))
    # = labels * ln(1 + exp(-y_pred)) + (1 - labels) * (-ln(exp(-y_pred))
    #   + ln(1 + exp(-y_pred)))
    # = labels * ln(1 + exp(-y_pred))
    #   + (1 - labels) * (y_pred + ln(1 + exp(-y_pred))
    # = (1 - labels) * y_pred + ln(1 + exp(-y_pred))
    # = y_pred - y_pred * labels + ln(1 + exp(-y_pred))

    # For y_pred < 0, to avoid overflow in exp(-y_pred), we reformulate the
    # above:
    # y_pred - y_pred * labels + ln(1 + exp(-y_pred))
    # = ln(exp(y_pred)) - y_pred * labels + ln(1 + exp(-y_pred))
    # = - y_pred * labels + ln(1 + exp(y_pred))

    # Hence, to ensure stability and avoid overflow, use this equivalent
    # formulation:
    # elementwise_cross_entropy
    # = may_pred(y_pred, 0) - y_pred * labels + ln(1 + exp(-abs(y_pred)))

    # Note, y_pred and labels must have the same shape.

    elementwise_cross_entropy = (
        np.maximum(y_pred, np.zeros(y_pred.shape))
        - y_pred * labels
        + np.log(1 + np.exp(-abs(y_pred)))
    )

  else:
    # p(y) = y_pred

    # elementwise_cross_entropy
    # = labels * -ln(y_pred) + (1 - labels) * -ln(1 - y_pred)

    elementwise_cross_entropy = -labels * np.log(y_pred) - (
        1 - labels
    ) * np.log(1 - y_pred)

  return np.mean(elementwise_cross_entropy)
