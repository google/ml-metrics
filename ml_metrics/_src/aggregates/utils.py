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
"""Utils for aggregates."""
from __future__ import annotations

import collections
import dataclasses

import chainable
from chainable import types
from ml_metrics._src.utils import math_utils


@dataclasses.dataclass
class MeanState(chainable.CallableMetric):
  """Mergeable states for batch update in an aggregate function."""

  total: types.NumbersT = 0.0
  count: types.NumbersT = 0

  def new(self, inputs: types.NumbersT) -> types.NumbersT:
    return MeanState(total=sum(inputs), count=len(inputs))

  def merge(self, other: MeanState):
    self.total += other.total
    self.count += other.count

  def result(self):
    return math_utils.safe_divide(self.total, self.count)


@dataclasses.dataclass
class TupleMeanState(chainable.CallableMetric):
  """MeanState for a tuple of inputs."""

  states: tuple[MeanState, ...] = ()

  def new(self, *inputs: tuple[types.NumbersT, ...]) -> TupleMeanState:
    return TupleMeanState(tuple(MeanState().new(x) for x in inputs))

  def merge(self, other: TupleMeanState):
    if not self.states:
      self.states = tuple(MeanState() for _ in other.states)
    for state, state_other in zip(self.states, other.states, strict=True):
      state.merge(state_other)

  def result(self):
    return tuple(state.result() for state in self.states)


@dataclasses.dataclass
class FrequencyState:
  """Mergeable frequency states for batch update in an aggregate function."""

  # TODO(b/331796958): Optimize storage consumption
  counter: collections.Counter[str] = dataclasses.field(
      default_factory=collections.Counter
  )
  count: int = 0

  def merge(self, other: 'FrequencyState'):
    self.counter.update(other.counter)
    self.count += other.count

  def result(self) -> list[tuple[str, float]]:
    result = [
        (key, math_utils.safe_divide(value, self.count))
        for key, value in self.counter.items()
    ]
    result = sorted(result, key=lambda x: (-x[1], x[0]))
    return result
