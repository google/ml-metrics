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
"""Math Utils."""

from ml_metrics._src import types
from ml_metrics._src.aggregates import types as agg_types
import numpy as np


def pos_sqrt(value) -> agg_types.NumbersT:
  """Returns sqrt of value or raises ValueError if negative."""
  if np.any(value < 0):
    raise ValueError('Attempt to take sqrt of negative value: {}'.format(value))
  return np.sqrt(value)


def safe_divide(a, b) -> agg_types.NumbersT:
  """Divide arguments element-wise (a / b), but returns zero(s) if b is 0."""
  result = np.divide(
      a, b, out=np.zeros_like(a, dtype=agg_types.DefaultDType), where=(b != 0)
  )

  return result.item() if result.ndim == 0 else result


def safe_to_scalar(arr: agg_types.NumbersT) -> agg_types.NumbersT:
  """Returns tuple, list, or np.ndarray as a scalar. Returns 0.0 if empty.

  Originally from tensorflow_model_analysis/metrics/metric_util.py

  Args:
    arr: A one element tuple, list, or numpy.ndarray to be converted to a
      scalar.

  Returns:
    The Python scalar.
  """
  if isinstance(arr, np.ndarray):
    if arr.size == 0:
      # 0 elements.
      return 0.0
    if arr.size == 1:
      # 1 element.
      return arr.item()
  else:
    # arr is tuple or list.
    if not arr:
      # 0 elements.
      return 0.0
    if len(arr) == 1:
      # 1 element.
      return arr[0]

  # >1 element.
  raise ValueError('Array should have exactly 1 value to a Python scalar')


def where(
    condition: agg_types.NumbersT, x: agg_types.NumbersT, y: agg_types.NumbersT
):
  """Like np.where, but directly returns itself if condition is scalar."""
  if types.is_array_like(condition):
    return np.where(condition, x, y)
  # If x or y is scalar, return it directly to keep its type (vs. np.where
  # that wraps it in a one-element array).
  return x if condition else y


def nanadd(a: agg_types.NumbersT, b: agg_types.NumbersT) -> agg_types.NumbersT:
  """Returns element-wise a + b, but ignores NaN as 0 unless both are NaN."""
  a_nan, b_nan = np.isnan(a), np.isnan(b)
  result = where(a_nan, 0, a) + where(b_nan, 0, b)
  # Revovers the NaN values if both operands are NaNs.
  return where(a_nan & b_nan, np.nan, result)
