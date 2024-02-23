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
"""Common types for the aggregates."""

import enum

from numpy import typing as npt

NumbersT = npt.ArrayLike
DefaultDType = float


# TODO: b/312290886 - move this to Python StrEnum when moved to Python 3.11.
class StrEnum(str, enum.Enum):
  """Enum where members also must be strings."""

  __str__ = str.__str__

  __repr__ = str.__repr__

  __format__ = str.__format__

  __iter__ = enum.Enum.__iter__


class InputType(StrEnum):  # pylint: disable=invalid-enum-extension
  """Label prediction encoding types."""

  # 1D array per batch, e.g., [0,1,0,1,0], [-1, 1, -1], or ['Y', 'N']
  BINARY = 'binary'
  # 1D array of floats typically is the probability for the binary
  # classification problem, e.g., [0.2, 0.3, 0.9]
  CONTINUOUS = 'continuous'
  # 2D array of the floats for the multilabel/multiclass classification problem.
  # Dimension: BatchSize x # Class
  # e.g., [[0.2, 0.8, 0.9], [0.1, 0.2, 0.7]].
  CONTINUOUS_MULTIOUTPUT = 'continuous-multioutput'
  # 1D array of class identifiers, e.g, ['a', 'b'] or [1, 29, 12].
  MULTICLASS = 'multiclass'
  # 2D lists of multiclass encodings of the classes, e.g., [[1,2,0], [3,2,0]]
  # The list can be ragged, e.g, [ ['a', 'b'], ['c'] ]
  MULTICLASS_MULTIOUTPUT = 'multiclass-multioutput'
  # 2D array of one-hot encoding of the classes, e.g., [[0,1,0], [0,0,1]]
  # This is a special case for "multilabel-indicator" except that only one
  # class is set to positive per example.
  MULTICLASS_INDICATOR = 'multiclass-indicator'


class AverageType(StrEnum):  # pylint: disable=invalid-enum-extension
  """Average type of the confusion matrix."""

  # Treats each class as one example and calculates the metrics on the total
  # aggregates of the result.
  MICRO = 'micro'
  # Macro calculates metrics for each class first, then average them across
  # classes.
  MACRO = 'macro'
  # Macro average with explicit weights per class.
  WEIGHTED = 'weighted'
  # Samples average calculates the metrics per example, and average them across
  # all examples.
  SAMPLES = 'samples'
  # Average for the positive label only.
  BINARY = 'binary'
