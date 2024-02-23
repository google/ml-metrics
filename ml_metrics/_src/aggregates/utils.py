# Copyright 2023 Google LLC
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
"""Utils for aggregates."""
import dataclasses
from ml_metrics._src.aggregates import types
import numpy as np


def safe_divide(a, b):
  result = np.divide(
      a, b, out=np.zeros_like(a, dtype=types.DefaultDType), where=(b != 0)
  )
  if result.ndim == 0:
    return result.item()
  return result


def pos_sqrt(value):
  """Returns sqrt of value or raises ValueError if negative."""
  if np.any(value < 0):
    raise ValueError('Attempt to take sqrt of negative value: {}'.format(value))
  return np.sqrt(value)


@dataclasses.dataclass
class MeanState:
  """Mergeable states for batch update in an aggregate function."""

  total: types.NumbersT = 0.0
  count: types.NumbersT = 0

  def __iadd__(self, other: 'MeanState'):
    self.total += other.total
    self.count += other.count
    return self

  def result(self):
    return safe_divide(self.total, self.count)
