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
from collections.abc import Callable, Iterable
import dataclasses
from typing import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from ml_metrics._src import types
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree

_T = TypeVar('_T')
_ResolvableOrMakeable = types.Resolvable[_T] | types.Makeable[_T]


@runtime_checkable
class Metric(Protocol):
  """MergibleMetric can be used as simple Map, and also MapReduce."""

  @abc.abstractmethod
  def add(self, *inputs, **named_inputs) -> Any:
    """Computes the state from a batch while outputting batch output."""

  @abc.abstractmethod
  def result(self) -> Any:
    """Returns the result of the metric."""

_MetricT = TypeVar('_MetricT', bound=Metric)


def as_agg_fn(
    cls: Callable[..., _MetricT],
    *args,
    nested: bool = False,
    agg_preprocess_fn: Callable[..., Any] | None = None,
    **kwargs,
) -> AggregateFn:
  """Creates an AggregateFn from a metric class."""
  deferred_metric: types.Resolvable[_MetricT] = lazy_fns.trace(cls)(
      *args, **kwargs
  )
  # Try resolve the target at construction at calltime to detect errors.
  _ = lazy_fns.maybe_make(deferred_metric)
  agg_fn = MergeableMetricAggFn(deferred_metric)
  if nested:
    agg_fn = AggFnNested(agg_fn, preprocess_fn=agg_preprocess_fn)
  return agg_fn


@runtime_checkable
class MergeableMetric(Metric, Protocol):
  """MergibleMetric can be used as simple Map, and also MapReduce."""

  @abc.abstractmethod
  def merge(self, other: Self):
    """Merges the metric with another metric of the same type."""


class CallableMetric(MergeableMetric, Callable[..., Any]):
  """A metric that is also callable.

  The CallableMetric is the recommended interface to implement a metric that
  supports both calculating batch result (`process`) and merging batch results
  (`merge`). A default `add` method is provided, but should be overwritten if
  the `merge` method is not applicable for `add`.
  """

  @abc.abstractmethod
  def new(self, *args, **kwargs) -> Self:
    """Calculate the suffient statistics, should be idemponent."""

  def add(self, *args, **kwargs):
    """Updates the sufficient statistics with a batch of inputs."""
    batch_result = self.new(*args, **kwargs)
    self.merge(batch_result)
    return batch_result

  def __call__(self, *args, **kwargs):
    """Calculates the result from the sufficient statistics."""
    return self.new(*args, **kwargs).result()


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
    if not (
        types.is_resolvable(metric_maker) or types.is_makeable(metric_maker)
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
    metric = self.metric_maker
    if types.is_makeable(metric):
      return metric.make()
    elif types.is_resolvable(metric):
      return metric.result_()
    else:
      raise TypeError(f'{type(metric)} is not a Makeable or Resolvable.')

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


class AggFnNested(AggregateFn):
  """AggregateFn that traverses and aggregates each leaf of a PyTree."""

  fn: Aggregatable
  preprocess_fn: Callable[..., tree.MapLikeTree[Any]] | None

  def __init__(
      self,
      fn: Aggregatable,
      preprocess_fn: Callable[..., tree.MapLikeTree[Any]] | None = None,
  ):
    if preprocess_fn is not None and not callable(preprocess_fn):
      raise ValueError(f'preporcess_fn must be a callable. got {preprocess_fn}')
    if not isinstance(fn, Aggregatable):
      raise ValueError(f'fn must be an instance of Aggregatable. got {fn}')
    super().__init__()
    self.fn = fn
    self.preprocess_fn = preprocess_fn

  def create_state(self):
    """Creates the initial states for the aggregation."""
    return None

  def update_state(
      self, state: tree.TreeMapView, inputs: tree.MapLikeTree[Any]
  ):
    """Update the state from a batch of inputs."""
    if self.preprocess_fn:
      inputs = self.preprocess_fn(inputs)
    if state is None:
      state = tree.TreeMapView(
          inputs, map_fn=lambda x: self.fn.create_state()
      ).apply()
      state = tree.TreeMapView(state)
    inputs = tree.TreeMapView.as_view(inputs)
    return state.copy_and_update(
        (k, self.fn.update_state(state[k], v)) for k, v in inputs.items()
    )

  # TODO: b/311207032 - Implement this.
  def merge_states(self, states):
    """Mering multiple states into a one state value."""
    raise NotImplementedError()

  def get_result(self, state):
    """Computes and returns the result from state."""
    return tree.TreeMapView.as_view(state, map_fn=self.fn.get_result).apply()


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
