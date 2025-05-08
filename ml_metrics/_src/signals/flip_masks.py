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
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np


@telemetry.WithTelemetry('ml_metrics', 'signals', 'binary_flip_mask')
def binary_flip_mask(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT | None = None,
) -> types.NumbersT:
  """AKA symmetric flip mask. Returns a 1 if the predictions don't match."""
  if threshold is not None:
    base_prediction = base_prediction > threshold
    model_prediction = model_prediction > threshold

  return np.logical_xor(base_prediction, model_prediction).astype(int)


@telemetry.WithTelemetry('ml_metrics', 'signals', 'neg_to_pos_flip_mask')
def neg_to_pos_flip_mask(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT | None = None,
) -> types.NumbersT:
  """Returns a 1 if base_prediction <= threshold < model_prediction."""
  if threshold is None:
    return not base_prediction and model_prediction

  base_under_threshold = base_prediction <= threshold
  model_over_threshold = model_prediction > threshold

  return np.logical_and(base_under_threshold, model_over_threshold).astype(int)


@telemetry.WithTelemetry('ml_metrics', 'signals', 'pos_to_neg_flip_mask')
def pos_to_neg_flip_mask(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT | None = None,
) -> types.NumbersT:
  """Returns a 1 if base_prediction > threshold >= model_prediction."""
  if threshold is None:
    return base_prediction and not model_prediction

  base_over_threshold = base_prediction > threshold
  model_under_threshold = model_prediction <= threshold

  return np.logical_and(base_over_threshold, model_under_threshold).astype(int)
