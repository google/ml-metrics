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
"""Flip Masks."""

from ml_metrics._src.aggregates import types
from ml_metrics.google.tools.signal_registry import registry
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def binary_flip_mask(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT | None = None,
) -> types.NumbersT:
  """Calculates the binary (symmetric) flip mask between two predictions.

  A flip occurs when the base prediction and the model prediction do not match.

  Args:
    base_prediction: The predictions from the base model.
    model_prediction: The predictions from the candidate model.
    threshold: Optional threshold to binarize predictions. If provided,
      predictions are converted to booleans (> threshold) before comparison.

  Returns:
    An integer or array of integers where 1 indicates a flip and 0 indicates no
    flip.

  Examples:
    >>> binary_flip_mask(np.array([0, 1]), np.array([1, 1]))
    array([1, 0])
    >>> binary_flip_mask(np.array([0.1, 0.6]), np.array([0.2, 0.4]),
    threshold=0.5)
    array([0, 1])
  """
  if threshold is not None:
    base_prediction = base_prediction > threshold
    model_prediction = model_prediction > threshold

  return np.logical_xor(base_prediction, model_prediction).astype(int)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def neg_to_pos_flip_mask(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT | None = None,
) -> types.NumbersT:
  """Calculates the negative-to-positive flip mask.

  A flip occurs when the base prediction is <= threshold and the model
  prediction is > threshold.

  Args:
    base_prediction: The predictions from the base model.
    model_prediction: The predictions from the candidate model.
    threshold: Optional threshold to binarize predictions. If None, predictions
      are assumed to be boolean scalars.

  Returns:
    An integer or array of integers where 1 indicates a negative-to-positive
    flip and 0 indicates no flip. If threshold is None and inputs are scalars,
    returns a boolean.

  Examples:
    >>> neg_to_pos_flip_mask(np.array([0.1, 0.6]), np.array([0.6, 0.4]),
    threshold=0.5)
    array([1, 0])
    >>> neg_to_pos_flip_mask(False, True)
    True
  """
  if threshold is None:
    return not base_prediction and model_prediction

  base_under_threshold = base_prediction <= threshold
  model_over_threshold = model_prediction > threshold

  return np.logical_and(base_under_threshold, model_over_threshold).astype(int)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def pos_to_neg_flip_mask(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT | None = None,
) -> types.NumbersT:
  """Calculates the positive-to-negative flip mask.

  A flip occurs when the base prediction is > threshold and the model
  prediction is <= threshold.

  Args:
    base_prediction: The predictions from the base model.
    model_prediction: The predictions from the candidate model.
    threshold: Optional threshold to binarize predictions. If None, predictions
      are assumed to be boolean scalars.

  Returns:
    An integer or array of integers where 1 indicates a positive-to-negative
    flip and 0 indicates no flip. If threshold is None and inputs are scalars,
    returns a boolean.

  Examples:
    >>> pos_to_neg_flip_mask(np.array([0.6, 0.1]), np.array([0.4, 0.6]),
    threshold=0.5)
    array([1, 0])
    >>> pos_to_neg_flip_mask(True, False)
    True
  """
  if threshold is None:
    return base_prediction and not model_prediction

  base_over_threshold = base_prediction > threshold
  model_under_threshold = model_prediction <= threshold

  return np.logical_and(base_over_threshold, model_under_threshold).astype(int)
