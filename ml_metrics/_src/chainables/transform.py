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
"""Transform Lib."""
from __future__ import annotations

import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any, Generic, TypeVar

from ml_metrics._src.aggregates import base as aggregates
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns


TreeMapKey = tree.TreeMapKey
TreeMapKeys = tree.TreeMapKeys

Fn = Callable[..., Any]
FnT = TypeVar('FnT', Callable, aggregates.Aggregatable)
TreeFnT = TypeVar('TreeFnT', bound=tree_fns.TreeFn)
TreeMapKeyT = TypeVar('TreeMapKeyT', bound=TreeMapKey)
Map = Mapping[TreeMapKeyT, Any]
TreeTransformT = TypeVar('TreeTransformT', bound='TreeTransform')
TreeFn = tree_fns.TreeFn


@dataclasses.dataclass
class ChainedTreeFn(tree.MapLikeTreeCallable):
  """Chaining the TreeFn together and runs it in sequence when called."""

  fns: Sequence[tree.MapLikeTreeCallable] = ()

  def __call__(
      self, inputs: tree.MapLikeTree | None = None
  ) -> tree.MapLikeTree | None:
    result = inputs
    for fn in self.fns:
      try:
        result = fn(result)
      except Exception as e:  # pylint: disable=too-broad-exception
        raise ValueError(f'Falied to execute {fn} with inputs: {result}') from e

    return result


@dataclasses.dataclass(frozen=True, kw_only=True)
class TreeTransform(Generic[TreeFnT]):
  """A lazy transform interface that works on a map like data.

  Each Transform represents a routine operating on a map-like data and returning
  a map-like result. There are following main operations suppoted:
    * Apply: it applies a routine on the inputs and directly returns the ouputs
      from the routine.
    * Assign: it applies a routine on the inputs and assign the result back to
      the inputs with the provided output_keys.
    * Select: it is a routineless Apply that selects some inputs given some
      input keys and optionally with the output_keys.
    * Aggregate: it applies aggregate function(s) on the inputs and outputs the
      aggregation results.

  note: the transform can be chainable by themselves by intializing with an
  input trasnform. `aggregate()` automatically separates the transform into two
  transforms by assigning the pre-aggregate transform as the input_transform
  of the new AggregateTransform constructed by calling `aggregate()`.

  The following is an example of running an image classification model and
  calculating corresponding metrics:
    predictions = (
        core.TreeTransform.new()
        # Example pre_processing
        .apply(
            output_keys=('image_bytes', 'label_id', Key.SKIP, 'label_text'),
            fn=parse_proto,
            input_keys='protos',
        )
        # Inference
        .assign(
            'predictions',
            fn=keras.models.load_model(model_path),
            input_keys='image_bytes',
        )
        # Example post_processing
        .assign(
            'pred_id',
            fn=lambda pred: str(np.argmax(pred)),
            input_keys='predictions',
        )
        .assign(
            'label_id',
            fn=str,
            input_keys='label_id',
        # Metrics.
        ).aggregate(
            input_keys=('label_id', 'pred_id'),
            fn=retrieval.TopKRetrievalAggFn(
                metrics=('accuracy', 'precision', 'recall', 'f1_score'),
                input_type=retrieval.InputType.MULTICLASS,
            ),
            output_keys=('accuracy', 'precision', 'recall', 'f1_score'),
        )
    )

  Attributes:
    input_transform: the transform that outputs the input of this transform.
    fns: the underlying routines of the transforms.
  """

  input_transform: TreeTransform | None = None
  fns: list[TreeFnT] = dataclasses.field(default_factory=list)
  _default_constructor: bool = True

  def __post_init__(self):
    if self._default_constructor:
      raise ValueError(
          f'Do not use default constructor {self.__class__.__name__}, uses'
          ' "new()" instead.'
      )

  @classmethod
  def new(cls, *, input_transform: TreeTransformT | None = None):
    if input_transform and not input_transform.fns:
      # Skip the input_transform if there is no routine to execute.
      input_transform = input_transform.input_transform
    return cls(input_transform=input_transform, _default_constructor=False)

  def make(self, *, recursive=True):
    fns = []
    if recursive and (input_transform := self.input_transform):
      fns = [input_transform.make()]
    fns += [tree_fn.maybe_make() for tree_fn in self.fns]
    assert fns, f'No function tp run in {self}, {fns=}'
    return ChainedTreeFn(fns=fns)

  def assign(
      self,
      output_keys: TreeMapKey | TreeMapKeys = (),
      *,
      fn: tree_fns.TreeFn | None = None,
      input_keys: TreeMapKey | TreeMapKeys = (),
  ) -> AssignTransform:
    return AssignTransform.new(input_transform=self).assign(
        output_keys,
        fn=fn,
        input_keys=input_keys,
    )

  def select(
      self, input_keys: TreeMapKeys, output_keys: TreeMapKeys | None = None
  ) -> TreeTransform:
    fn = tree_fns.Select.new(input_keys=input_keys, output_keys=output_keys)
    return dataclasses.replace(self, fns=self.fns + [fn])

  def aggregate(self, **kwargs) -> AggregateTransform:
    # TODO: b/318463291 - fix a bug when self.fns is empty, the runner
    # skips its own input transform.
    return AggregateTransform.new(input_transform=self).add_aggregate(**kwargs)

  def agg(self, **kwargs) -> AggregateTransform:
    """Alias for aggregate."""
    return self.aggregate(**kwargs)

  def apply(
      self,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn: tree_fns.TreeFn | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> TreeTransformT:
    """Applys a TreeFn on the selected inputs and directly outputs the result."""
    fn = tree_fns.TreeFn.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    return dataclasses.replace(self, fns=self.fns + [fn])


def make(tree_transform: TreeTransform):
  """Makes the runnable instance from transfrom."""
  return tree_transform.make()


class AssignTransform(TreeTransform):
  """A MapTransform captures all transforms that have row correspondence."""

  def assign(
      self,
      output_keys: TreeMapKey | TreeMapKeys = (),
      *,
      fn: tree_fns.TreeFn | None = None,
      input_keys: TreeMapKey | TreeMapKeys = (),
  ) -> AssignTransform:
    """Assign some key value pairs back to the input mapping."""
    fn = tree_fns.Assign.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    return dataclasses.replace(self, fns=self.fns + [fn])


@dataclasses.dataclass(frozen=True, kw_only=True)
class StackedTreeAggregateFn(aggregates.AggregateFn):
  """Combining multile TreeAggregateFns into one TreeAggregateFn."""

  agg_fns: dict[tree_fns.TreeAggregateFn, tree_fns.TreeAggregateFn] = (
      dataclasses.field(default_factory=dict)
  )
  input_fn: tree_fns.TreeFn | None = None

  def create_state(self):
    return {
        key: tree_fn.create_state() for key, tree_fn in self.agg_fns.items()
    }

  def update_state(self, state: Any, inputs: Any) -> Any:
    if self.input_fn:
      inputs = self.input_fn(inputs)
    # All aggregates operate on the same inputs.
    for key, tree_fn in self.agg_fns.items():
      try:
        state[key] = tree_fn.update_state(state[key], inputs)
      except KeyError as e:
        raise KeyError(f'{key=} not found from: {list(state.keys())=}') from e
      except Exception as e:
        raise ValueError(
            f'Falied to update {tree_fn=} with inputs: {inputs=}'
        ) from e
    return state

  def merge_states(self, states: Any) -> Any:
    states_by_fn = collections.defaultdict(list)
    for state in states:
      for key, fn_state in state.items():
        states_by_fn[key].append(fn_state)
    return {
        key: self.agg_fns[key].merge_states(fn_states)
        for key, fn_states in states_by_fn.items()
    }

  def get_result(self, state: Any) -> tree.MapLikeTree[Any]:
    result = tree.MappingView()
    for key, fn_state in state.items():
      fn_result = self.agg_fns[key].get_result(fn_state)
      result = result | tree.MappingView.as_view(fn_result)
    return result.data


class AggregateTransform(TreeTransform[tree_fns.TreeAggregateFn]):
  """An AggregateTransform reduce rows."""

  def make(self, *, recursive: bool = True) -> StackedTreeAggregateFn:
    input_fn = None
    if recursive and (input_transform := self.input_transform):
      input_fn = input_transform.make()
    agg_fns = {}
    for tree_fn in self.fns:
      actual_tree_fn = tree_fn.maybe_make()
      if not isinstance(actual_tree_fn, tree_fns.TreeAggregateFn):
        raise ValueError(
            'Unexpected tree_fn type:'
            f'{type(actual_tree_fn)} for {actual_tree_fn=}'
        )
      agg_fns[tree_fn] = actual_tree_fn

    return StackedTreeAggregateFn(agg_fns=agg_fns, input_fn=input_fn)

  def add_aggregate(
      self,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn: aggregates.Aggregatable | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ):
    """Adds a aggregate and stack it on the existing aggregates."""
    fn = tree_fns.TreeAggregateFn.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    return dataclasses.replace(self, fns=self.fns + [fn])

# Register the lazy_fns so the make() can be called automatically when calling.
# lazy_fns.maybe_make(transform).
lazy_fns.makeables.register(TreeTransform, make)
lazy_fns.makeables.register(AssignTransform, make)
lazy_fns.makeables.register(AggregateTransform, make)
