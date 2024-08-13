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

from typing import Any

from ml_metrics._src.aggregates import types
import numpy as np


def pos_sqrt(value):
  """Returns sqrt of value or raises ValueError if negative."""
  if np.any(value < 0):
    raise ValueError('Attempt to take sqrt of negative value: {}'.format(value))
  return np.sqrt(value)


def safe_divide(x1, x2, k_epsilon=0):
  """Divide arguments element-wise (x1 / x2).

  Returns zero(s) if abs(x2) <= k_epsilon.

  Args:
    x1: Divident array.
    x2: Divisor array.
    k_epsilon: The minimum value of abs(x2) to divide by.

  Returns:
    The quotient x1 / x2, element-wise. This is a scalar if both x1 and x2 are
    scalars.
  """
  result = np.divide(
      x1,
      x2,
      out=np.zeros_like(x1, dtype=types.DefaultDType),
      where=(np.abs(x2) > k_epsilon),
  )

  return result.item() if result.ndim == 0 else result


def safe_to_scalar(arr: tuple[Any] | list[Any] | np.ndarray) -> Any:  # pylint: disable=g-one-element-tuple
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
