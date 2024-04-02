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
"""Example Count Metric."""

from collections.abc import Iterable
from typing import Any

from ml_metrics._src.aggregates import base


class ExampleCountAggFn(base.AggregateFn):
  """Computes the Example Count Metric."""

  def create_state(self) -> float:
    return 0.0

  def update_state(
      self,
      state: float,
      examples: Iterable[Any],
      example_weights: Iterable[float] | None = None,
  ) -> float:
    if example_weights is None:
      # Unweighted.
      return state + sum(1 for _ in examples)

    if sum(1 for _ in examples) != sum(1 for _ in example_weights):
      raise ValueError(
          'examples and example_weights must have the same length, but'
          f' recieved examples={examples} and'
          f' example_weights={example_weights}.'
      )

    return state + sum(example_weight for example_weight in example_weights)

  def merge_states(self, states: Iterable[float]) -> float:
    return sum(state for state in states)

  def get_result(self, state: float) -> float:
    return state
