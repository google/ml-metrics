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
"""Module for cross entropy loss functions."""

from ml_metrics._src.aggregates import types
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np


def _check_y_true_contains_only_0_and_1(y_true: types.NumbersT) -> None:
  if not all(y == 0 or y == 1 for y in y_true):
    raise ValueError(
        'y_true must contain only 0s and 1s, but recieved: {}'.format(y_true)
    )


@telemetry.function_monitor(
    api='ml_metrics', category=telemetry.CATEGORY.SIGNAL
)
def binary_cross_entropy(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
) -> float:
  """Calculates binary cross entropy loss for two lists of labels.

  Args:
    y_true: Truth label. This is either 0 or 1.
    y_pred: Predicted value. This is the model's prediction, i.e, a single
      floating-point value which represents a probability (i.e., value in (0.,
      1.)).

  Returns:
    The binary cross-entropy loss between true labels and predicted labels.
  """
  _check_y_true_contains_only_0_and_1(y_true)

  return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


@telemetry.function_monitor(
    api='ml_metrics', category=telemetry.CATEGORY.SIGNAL
)
def categorical_cross_entropy(
    y_true: types.NumbersT,
    y_pred: types.NumbersT,
):
  """Calculates categorical cross entropy loss for two lists of labels.

  Args:
    y_true: Truth label. This is either 0 or 1.
    y_pred: Predicted value. This is the model's prediction, i.e, a single
      floating-point value which represents a probability (i.e., value in [0.,
      1.]).

  Returns:
    The categorical cross-entropy loss between true labels and predicted labels.
  """
  _check_y_true_contains_only_0_and_1(y_true)

  return -np.sum(y_true * np.log(y_pred / np.sum(y_pred)))
