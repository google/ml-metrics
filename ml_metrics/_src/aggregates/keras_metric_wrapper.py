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
"""Keras metric wrapper."""
from __future__ import annotations

from collections.abc import Iterable
import dataclasses
from typing import Any, Protocol

from ml_metrics._src.aggregates import base as agg


class KerasMetric(Protocol):
  """Base interface for Keras metrics."""

  def update_state(self, *inputs, **named_inputs) -> None:
    """Updates the state from a batch of inputs."""

  def reset_state(self) -> None:
    """Resets the state."""

  def merge_state(self, other: Iterable[KerasMetric]) -> None:
    """Merges the state with another state of the same type."""

  def result(self) -> Any:
    """Returns the result of the metric."""


def is_keras_metric(metric: Any) -> bool:
  """Duck type check for Keras metric."""
  return (
      hasattr(metric, "update_state")
      and hasattr(metric, "reset_state")
      and hasattr(metric, "merge_state")
      and hasattr(metric, "result")
  )


@dataclasses.dataclass
class KerasAggregateFn(agg.AggregateFn):
  """AggregateFn for Keras metrics."""

  metric: KerasMetric

  def __post_init__(self):
    if is_keras_metric(self.metric):
      self.metric.reset_state()
      self._metric = self.metric
    else:
      try:
        assert hasattr(self.metric, "__call__")
        self._metric = self.metric()
        if not is_keras_metric(self._metric):
          raise TypeError("metric must implement Keras metric base interface.")
      except Exception as e:
        raise TypeError(
            f"Cannot construct a Keras metric from {self.metric}."
        ) from e

  def create_state(self) -> KerasMetric:
    assert hasattr(self._metric, "reset_state")
    self._metric.reset_state()
    return self._metric

  def update_state(
      self, state: KerasMetric, *inputs: Any, **named_inputs: Any
  ) -> KerasMetric:
    state.update_state(*inputs, **named_inputs)
    return state

  def merge_states(self, states: Iterable[KerasMetric]) -> KerasMetric:
    # This in-place merges all the states into the first state and returns it.
    iter_states = iter(states)
    result = next(iter_states)
    result.merge_state(list(iter_states))
    return result

  def get_result(self, state: KerasMetric) -> Any:
    result = state.result()
    if hasattr(result, "numpy"):
      return result.numpy()
    return result
