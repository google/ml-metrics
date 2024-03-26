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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import dataclasses
import inspect
import itertools
import time
from typing import Any, Generic, TypeVar

from absl import logging
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


class PrefetchableIterator:
  """An iterator that can also prefetch before iterated."""

  def __init__(self, generator, prefetch_size: int = 2):
    self._data = []
    if not inspect.isgenerator(generator):
      generator = iter(generator)
    self._generator = generator
    self._prefetch_size = prefetch_size
    self._exhausted = False
    self.to_be_deleted = False
    self._error_cnt = 0
    self._cnt = 0

  def __next__(self):
    self.prefetch(1)
    if self._data:
      return self._data.pop(0)
    else:
      self.to_be_deleted = True
      raise StopIteration

  def __iter__(self):
    return self

  def prefetch(self, num_items: int = 0):
    """Prefeches items from the undelrying generator."""
    exhausted = False
    while not exhausted and len(self._data) < (
        num_items or self._prefetch_size
    ):
      try:
        self._data.append(next(self._generator))
        self._cnt += 1
      except StopIteration:
        exhausted = True
      except ValueError as e:
        logging.warning('Got error during prefetch: %s', e)
        self._error_cnt += 1
        if self._error_cnt > 3:
          time.sleep(1)
          break


def _call_fns(
    fns: Sequence[tree.MapLikeTreeCallable | tree_fns.TreeAggregateFn],
    inputs: tree.MapLikeTree | None = None,
) -> tree.MapLikeTree | None:
  result = inputs
  for fn in fns:
    try:
      result = fn(result)
    except Exception as e:  # pylint: disable=too-broad-exception
      raise ValueError(f'Falied to execute {fn} with inputs: {result}') from e

  return result


@dataclasses.dataclass(frozen=True, kw_only=True)
class CombinedTreeFn:
  """Combining multiple transforms into one concrete TreeFn."""

  input_fns: Sequence[tree.MapLikeTreeCallable] = ()
  agg_fns: dict[tree_fns.TreeAggregateFn, aggregates.Aggregatable] = (
      dataclasses.field(default_factory=dict)
  )
  output_fns: Sequence[tree.MapLikeTreeCallable] = ()
  input_iterator: Iterable[Any] | None = None

  @classmethod
  def from_transform(
      cls,
      transform: TreeTransform,
      recursive: bool = True,
      aggregator: bool = False,
  ):
    """Builds a TreeFn from a transform and optionally its input transforms."""
    # Flatten the transform into nodes and find the first aggregation node as
    # the aggregation node for the concrete function. Any node before it is
    # input nodes, any node after it regardless of its type (aggregation or not)
    # is output_node.
    transforms = transform.flatten_transform() if recursive else [transform]
    input_nodes, output_nodes = [], []
    iterator_node, agg_node = None, None
    for node in transforms:
      if isinstance(node, IteratorSource):
        iterator_node = node
      elif agg_node is None and isinstance(node, AggregateTransform):
        agg_node = node
      elif agg_node is None:
        input_nodes.append(node)
      else:
        output_nodes.append(node)
    # Reset the iterator_node and input_nodes if this is an aggregator.
    if aggregator:
      iterator_node = None
      input_nodes = []
    # Collect input_iterator.
    input_iterator = lazy_fns.maybe_make(
        iterator_node and iterator_node.iterator
    )
    # Collect all the input functions from the input nodes.
    input_fns = list(
        itertools.chain.from_iterable(node.fns for node in input_nodes)
    )
    input_fns = [tree_fn.maybe_make() for tree_fn in input_fns]
    # Collect all the aggregation functions from the aggregation node.
    agg_fns = {}
    if agg_node:
      for agg_fn in agg_node.fns:
        actual_agg_fn = agg_fn.maybe_make()
        agg_fns[agg_fn] = actual_agg_fn
        if not isinstance(actual_agg_fn, aggregates.Aggregatable):
          raise ValueError(f'Not an aggregatable: {agg_fn}: {actual_agg_fn}')
    # Collect all the output functions from the output nodes.
    output_fns = []
    for node in output_nodes:
      # the output nodes can be aggregate, uses it as a callable since they are
      # all ran in-process (after the first aggregation node).
      if isinstance(node, AggregateTransform):
        output_fns.append(cls.from_transform(node, recursive=False))
      else:
        output_fns.extend([tree_fn.maybe_make() for tree_fn in node.fns])
    return cls(
        input_fns=input_fns,
        agg_fns=agg_fns,
        output_fns=output_fns,
        input_iterator=input_iterator,
    )

  def create_state(self):
    return {
        key: tree_fn.create_state() for key, tree_fn in self.agg_fns.items()
    }

  def _update_state(self, state, inputs):
    """Updates the state by inputs."""
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

  def get_result(self, state: Any) -> tree.MapLikeTree[Any] | None:
    result = tree.MappingView()
    for key, fn_state in state.items():
      fn_result = self.agg_fns[key].get_result(fn_state)
      result = result | tree.MappingView.as_view(fn_result)
    result = result.data
    return _call_fns(self.output_fns, result)

  def _actual_inputs(self, inputs, input_iterator):
    if inputs is not None and input_iterator:
      raise ValueError(f'Cannot set inputs and input_iterator from: {self}')
    # Overrides the internal input_iterator if either inputs or input_iterator
    # is provided.
    if not input_iterator and not self.input_iterator:
      return [inputs]
    else:
      return input_iterator or self.input_iterator

  def iter_call(self, input_iterator: Iterable[Any] = ()) -> Iterator[Any]:
    """Directly apply aggregate on inputs."""
    input_iterator = self._actual_inputs(None, input_iterator)
    for batch in input_iterator:
      yield _call_fns(self.input_fns, batch)

  def update_state(
      self,
      state: Any = None,
      inputs=None,
      *,
      input_iterator: Iterable[Any] = (),
  ):
    """Updates the state by either the inputs or an iterator of the inputs."""
    input_iterator = self._actual_inputs(inputs, input_iterator)
    state = state or self.create_state()
    for batch_output in self.iter_call(input_iterator):
      state = self._update_state(state, batch_output)
    return state

  def __call__(self, inputs=None, *, input_iterator=()):
    iter_input = self._actual_inputs(inputs, input_iterator)
    if self.agg_fns:
      return self.get_result(self.update_state(input_iterator=iter_input))
    else:
      result = list(self.iter_call(iter_input))
      # Directly returns the result when inputs (vs. iterator) is fed.
      if not input_iterator and not self.input_iterator:
        return result[0]
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
        # Reading
        .data_source(iterator_make=get_iterator(path))
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
  _default_constructor: bool = dataclasses.field(default=True, repr=False)

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

  def chain(self, transform: TreeTransform):
    """Chains self to the input of the first node in the other transform."""
    transforms = transform.flatten_transform()
    input_transform = self
    for transform in transforms:
      input_transform = dataclasses.replace(
          transform, input_transform=input_transform
      )
    return input_transform

  def make(self, *, recursive=True, aggregator=False):
    """Makes the concrete function instance from the transform."""
    return CombinedTreeFn.from_transform(
        self, recursive=recursive, aggregator=aggregator
    )

  def data_source(self, iterator: Any = None):
    if self.input_transform:
      raise ValueError(
          f'Cannot add data_source to {self} when there is already an'
          ' input_transform.'
      )
    return IteratorSource.new(iterator)

  def assign(
      self,
      output_keys: TreeMapKey | TreeMapKeys = (),
      *,
      fn: lazy_fns.Makeable | Callable[..., Any] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> TreeTransform:
    """Assign some key value pairs back to the input mapping."""
    fn = tree_fns.Assign.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    return self._maybe_new_transform(fn)

  def select(
      self, input_keys: TreeMapKeys, output_keys: TreeMapKeys | None = None
  ) -> TreeTransform:
    output_keys = output_keys or input_keys
    fn = tree_fns.Select.new(input_keys=input_keys, output_keys=output_keys)
    return self._maybe_new_transform(fn)

  def aggregate(
      self,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      *,
      fn: lazy_fns.Makeable | aggregates.Aggregatable | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> AggregateTransform:
    # TODO: b/318463291 - fix a bug when self.fns is empty, the runner
    # skips its own input transform.
    return AggregateTransform.new(input_transform=self).add_aggregate(
        input_keys=input_keys, output_keys=output_keys, fn=fn
    )

  def agg(self, **kwargs) -> AggregateTransform:
    """Alias for aggregate."""
    return self.aggregate(**kwargs)

  def apply(
      self,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn: lazy_fns.Makeable | Callable[..., Any] | None = None,
      input_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
  ) -> TreeTransformT:
    """Applys a TreeFn on the selected inputs and directly outputs the result."""
    fn = tree_fns.TreeFn.new(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
    )
    return self._maybe_new_transform(fn)

  def flatten_transform(self):
    """Flatten all the chain of transforms into a list."""
    if not self.input_transform:
      return [self]
    return self.input_transform.flatten_transform() + [self]

  def _maybe_new_transform(self, fn):
    # Break apart the transform when this is an aggregate transform.
    if isinstance(self, AggregateTransform):
      return TreeTransform(
          input_transform=self, fns=[fn], _default_constructor=False
      )
    elif isinstance(self, IteratorSource):
      return TreeTransform(
          input_transform=self, fns=[fn], _default_constructor=False
      )
    else:
      return dataclasses.replace(self, fns=self.fns + [fn])


def make(tree_transform: TreeTransform):
  """Makes the runnable instance from transfrom."""
  return tree_transform.make()


@dataclasses.dataclass(frozen=True, kw_only=True)
class IteratorSource(TreeTransform):
  """A source that wraps around an iterator."""

  # Any iterable or an instance that can make an iterable either from the
  # registered lazy_fns.makeables or from an implemented Makeable.
  iterator: Iterable[Any] | lazy_fns.Makeable | Any

  def __post_init__(self):
    if self.input_transform:
      raise ValueError(
          f'Source cannot have an input_transform, got {self.input_transform}'
      )
    if self.fns:
      raise ValueError(f'Source must not have any fn, got {self.fns}')

  @classmethod
  def new(cls, iterator: Any):
    return cls(iterator=iterator, _default_constructor=False)


class AggregateTransform(TreeTransform[tree_fns.TreeAggregateFn]):
  """An AggregateTransform reduce rows.

  Attributes:
    input_transform: the transform that outputs the input of this transform.
    fns: the underlying routines of the transforms.
  """

  def add_aggregate(
      self,
      *,
      output_keys: TreeMapKey | TreeMapKeys = tree.Key.SELF,
      fn: lazy_fns.Makeable | aggregates.Aggregatable | None = None,
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
lazy_fns.makeables.register(TreeTransform, make)
lazy_fns.makeables.register(AggregateTransform, make)
