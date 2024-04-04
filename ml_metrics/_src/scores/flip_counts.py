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
"""Flip Counts."""

from ml_metrics._src.aggregates import types
import numpy as np

# TODO: b/333106326 - Replace flip_counts with flip_mask.


def flip_counts(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT = np.array(0.5),
) -> types.NumbersT:
  """AKA symmetric flip counts. Returns a 1 if the predictions don't match."""
  base_over_threshold = base_prediction > threshold
  model_over_threshold = model_prediction > threshold

  return np.logical_xor(base_over_threshold, model_over_threshold).astype(int)


def neg_to_neg_flip_counts(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT = np.array(0.5),
) -> types.NumbersT:
  """Returns a 1 if both predictions are less than or equal to the threshold."""
  base_under_threshold = base_prediction <= threshold
  model_under_threshold = model_prediction <= threshold

  return np.logical_and(base_under_threshold, model_under_threshold).astype(int)


def neg_to_pos_flip_counts(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT = np.array(0.5),
) -> types.NumbersT:
  """Returns a 1 if base_prediction <= threshold < model_prediction."""
  base_under_threshold = base_prediction <= threshold
  model_over_threshold = model_prediction > threshold

  return np.logical_and(base_under_threshold, model_over_threshold).astype(int)


def pos_to_neg_flip_counts(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT = np.array(0.5),
) -> types.NumbersT:
  """Returns a 1 if base_prediction > threshold >= model_prediction."""
  base_over_threshold = base_prediction > threshold
  model_under_threshold = model_prediction <= threshold

  return np.logical_and(base_over_threshold, model_under_threshold).astype(int)


def pos_to_pos_flip_counts(
    base_prediction: types.NumbersT,
    model_prediction: types.NumbersT,
    threshold: types.NumbersT = np.array(0.5),
) -> types.NumbersT:
  """Returns a 1 if both predictions are greater than the threshold."""
  base_over_threshold = base_prediction > threshold
  model_over_threshold = model_prediction > threshold

  return np.logical_and(base_over_threshold, model_over_threshold).astype(int)
