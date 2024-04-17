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

from ml_metrics._src.aggregates import types
import numpy as np


def pos_sqrt(value):
  """Returns sqrt of value or raises ValueError if negative."""
  if np.any(value < 0):
    raise ValueError('Attempt to take sqrt of negative value: {}'.format(value))
  return np.sqrt(value)


def safe_divide(a, b):
  """Divide arguments element-wise (a / b), but returns zero(s) if b is 0."""
  result = np.divide(
      a, b, out=np.zeros_like(a, dtype=types.DefaultDType), where=(b != 0)
  )

  return result.item() if result.ndim == 0 else result
