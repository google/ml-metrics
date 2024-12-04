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
"""Base AggregateFn for all the aggregates."""
from __future__ import annotations

import abc
from collections.abc import Iterable
import dataclasses
from typing import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from ml_metrics._src import base_types

_T = TypeVar('_T')
_ResolvableOrMakeable = base_types.Resolvable[_T] | base_types.Makeable[_T]


@runtime_checkable
class Metric(Protocol):
  """MergibleMetric can be used as simple Map, and also MapReduce."""

  @abc.abstractmethod
  def add(self, *inputs, **named_inputs) -> Any:
    """Computes the state from a batch while outputting batch output."""

  @abc.abstractmethod
  def result(self) -> Any:
    """Returns the result of the metric."""


@runtime_checkable
class MergeableMetric(Metric, Protocol):
  """MergibleMetric can be used as simple Map, and also MapReduce."""

  @abc.abstractmethod
  def merge(self, other: Self):
    """Merges the metric with another metric of the same type."""

_MetricT = TypeVar('_MetricT', bound=MergeableMetric)


@runtime_checkable
class Aggregatable(Protocol):
  """An aggregation interface, similar to apche_beam.CombineFn."""

  def create_state(self) -> Any:
    """Creates the initial states for the aggregation."""
    return None

  @abc.abstractmethod
  def update_state(self, state, *inputs, **named_inputs):
    """Update the state from a batch of inputs.

    Args:
      state: the current state.
      *inputs: elements to add.
      **named_inputs: elements to add.
    """

  def merge_states(self, states):
    """Mering multiple states into a one state value.

    This is only required for distributed implementations such as Beam. Only the
    first state may be modified and returned for efficiency.

    Args:
      states: the states to be merged.
    """
    raise NotImplementedError()

  def get_result(self, state):
    """Computes and returns the result from state.

    Args:
      state: the final state value computed by this CombineFn.

    Returns:
      state.
    """
    return state


class AggregateFn(Aggregatable):
  """An aggregation interface, similar to apche_beam.CombineFn."""

  def __call__(self, *inputs, **named_inputs):
    """Directly apply aggregate on inputs."""
    return self.get_result(
        self.update_state(self.create_state(), *inputs, **named_inputs)
    )


class MergeableMetricAggFn(AggregateFn, Generic[_MetricT]):
  """A aggregation wrapper for MergeableMetric."""

  metric_maker: _ResolvableOrMakeable[_MetricT]

  def __init__(self, metric_maker: _ResolvableOrMakeable[_MetricT]):
    super().__init__()
    if not isinstance(
        metric_maker, (base_types.Resolvable, base_types.Makeable)
    ):
      raise TypeError(
          'metric_maker must be an instance of Makeable or Resolvable. got'
          f' {type(metric_maker)}'
      )
    self.metric_maker = metric_maker

  def __eq__(self, other, /):
    return (
        isinstance(other, MergeableMetricAggFn)
        and self.metric_maker == other.metric_maker
    )

  def create_state(self) -> _MetricT:
    if isinstance(self.metric_maker, base_types.Resolvable):
      return self.metric_maker.result_()
    return self.metric_maker.make()

  def update_state(self, state: _MetricT, *args, **kwargs) -> _MetricT:
    state.add(*args, **kwargs)
    return state

  def merge_states(self, states: Iterable[_MetricT]) -> _MetricT:
    iter_states = iter(states)
    result = next(iter_states)
    for state in iter_states:
      result.merge(state)
    return result

  def get_result(self, state: _MetricT) -> Any:
    return state.result()


@dataclasses.dataclass(frozen=True)
class UserAggregateFn(AggregateFn):
  """An aggregation interface, similar to apche_beam.CombineFn."""

  fn: Aggregatable

  def __post_init__(self):
    if not isinstance(self.fn, Aggregatable):
      raise ValueError(
          f'UserAggregateFn must be an instance of Aggregatable. got {self.fn}'
      )

  def create_state(self) -> Any:
    """Creates the initial states for the aggregation."""
    return self.fn.create_state()

  def update_state(self, state, *inputs, **named_inputs):
    """Update the state from a batch of inputs."""
    return self.fn.update_state(state, *inputs, **named_inputs)

  def merge_states(self, states):
    """Mering multiple states into a one state value."""
    return self.fn.merge_states(states)

  def get_result(self, state):
    """Computes and returns the result from state."""
    return self.fn.get_result(state)
