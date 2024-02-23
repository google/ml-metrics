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
"""Tree functions."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, Generic, TypeVar

from ml_metrics._src.aggregates import base as aggregates
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree

ValueT = TypeVar('ValueT')
FnT = TypeVar('FnT')
StateT = TypeVar('StateT')
ValueT = TypeVar('ValueT')


# TODO: b/318463291 - Consider adding `inputs` fields to handle hybrid args.
@dataclasses.dataclass(kw_only=True, frozen=True)
class TreeFn(Generic[FnT, ValueT], tree.MapLikeTreeCallable[ValueT]):
  """A lazy function that takes a dict and outputs either a dict or a tuple.

  Attributes:
    output_keys: The output keys of the function outputs. If left empty
      (default), output will be only positional (a tuple).
    input_keys: The input keys used to take from the input dict. If left empty
      (default), this outputs an empty tuple.
    fn: A callable instance that takes positional inputs and outputs positional
      value (tuple if there are multiple).
    lazy: If True, the underlying function is lazy, normally, this means it
      needs to be constructed at runtime.
    id: A string that serves as an identifier for the TreeFn instance.
  """

  input_keys: tree.TreeMapKey | tree.TreeMapKeys | None = ()
  output_keys: tree.TreeMapKey | tree.TreeMapKeys = ()
  fn: FnT | lazy_fns.LazyFn[FnT] | None = None
  _default_constructor: bool = True

  def __post_init__(self):
    if self._default_constructor:
      raise ValueError(
          f'Do not use the constructor, use {self.__class__.__name__}.new().'
      )

  @classmethod
  def new(
      cls,
      *,
      output_keys: tree.TreeMapKey | tree.TreeMapKeys = tree.Key.SELF,
      fn: FnT | lazy_fns.LazyFn[FnT] | None = None,
      input_keys: tree.TreeMapKey | tree.TreeMapKeys = tree.Key.SELF,
  ) -> TreeFn[FnT, ValueT]:
    """Normalizes the arguements before constructing a TreeFn."""
    if fn:
      # Fn requires a tuple for positional inputs. Normalize_keys converts
      # the keys into a tuple of keys to make sure the actual selected inputs
      # are also wrapped in a tuple.
      input_keys = tree.normalize_keys(input_keys)
    if output_keys is not tree.Key.SELF:
      output_keys = tree.normalize_keys(output_keys)
    return cls(
        output_keys=output_keys,
        input_keys=input_keys,
        fn=fn,
        _default_constructor=False,
    )

  @property
  def lazy(self):
    return isinstance(self.fn, lazy_fns.LazyFn)

  @functools.cached_property
  def actual_fn(self) -> FnT:
    if self.lazy:
      # self.fn is always LazyFn if self.lazy is True.
      return lazy_fns.maybe_make(typing.cast(lazy_fns.LazyFn, self.fn))
    return self.fn

  def maybe_make(self: TreeFn) -> TreeFn:
    """Explicitly instantiate the lazy_fn of a tree_fn."""
    if self.lazy:
      return dataclasses.replace(self, fn=self.actual_fn)
    return self

  def get_inputs(self, inputs: tree.MapLikeTree[ValueT]) -> tuple[ValueT, ...]:
    # The input_keys are always normalized to tuple, so the output will always
    # be a tuple.
    fn_inputs = typing.cast(
        tuple[Any, ...], tree.MappingView.as_view(inputs)[self.input_keys]
    )
    return fn_inputs

  def __call__(
      self, inputs: tree.MapLikeTree[ValueT] | None = None
  ) -> tree.MapLikeTree[ValueT]:
    fn_inputs = self.get_inputs(inputs)
    if (fn := self.actual_fn) is not None:
      try:
        outputs = fn(*fn_inputs)
      except Exception as e:
        raise ValueError(f'Failed to call {self.fn} with {fn_inputs=}') from e
    else:
      # If the function is None, this serves as a select operation.
      outputs = fn_inputs
    # Uses MappingView to handle multiple key.
    return tree.MappingView().copy_and_set(self.output_keys, outputs).data

  def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('actual_fn', None)
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)


class Assign(TreeFn):
  """A lazy Map operation that operates on an mappable."""

  def __post_init__(self):
    super().__post_init__()
    if not self.output_keys:
      raise ValueError(
          f'Assign should have output_keys, got {self.output_keys=}'
      )

  def __call__(
      self, inputs: tree.MapLikeTree | None = None
  ) -> tree.MapLikeTree:
    fn_inputs = self.get_inputs(inputs)
    if (fn := self.actual_fn) is not None:
      try:
        outputs = fn(*fn_inputs)
      except Exception as e:
        raise ValueError(f'Failed to call {self.fn} with {fn_inputs=}') from e
    else:
      # If the function is None, this serves as a select operation.
      outputs = fn_inputs
    # Uses MappingView to handle multiple key.
    return tree.MappingView(inputs).copy_and_set(self.output_keys, outputs).data


class Select(TreeFn):
  """A lazy Map operation that operates on an mappable."""

  def __post_init__(self):
    assert self.fn is None, f'Select should have no fn, got {self.fn}.'


@dataclasses.dataclass(kw_only=True, frozen=True)
class TreeAggregateFn(
    Generic[StateT, ValueT], TreeFn[aggregates.Aggregatable, ValueT]
):
  """Transform with one AggregateFn with its input and output keys."""

  fn: (
      aggregates.Aggregatable | lazy_fns.LazyFn[aggregates.Aggregatable] | None
  ) = None

  # TODO: b/318463291 - adds groupby functionality.
  def create_state(self) -> StateT:
    try:
      return self.actual_fn.create_state()
    except Exception as e:
      raise ValueError(f'Cannot create_state in {self=}') from e

  def update_state(
      self, state: StateT, inputs: tree.MapLikeTree[ValueT]
  ) -> StateT:
    try:
      inputs = tree.MappingView.as_view(inputs)[self.input_keys]
      state = self.actual_fn.update_state(state, *inputs)
    except Exception as e:
      raise ValueError(f'Cannot call {self=} with {inputs=}') from e
    return state

  def merge_states(self, states: StateT) -> StateT:
    return self.actual_fn.merge_states(states)

  def get_result(self, state: StateT) -> tree.MapLikeTree[ValueT]:
    outputs = self.actual_fn.get_result(state)
    return tree.MappingView().copy_and_set(self.output_keys, outputs).data

  def __call__(
      self, inputs: tree.MapLikeTree[ValueT] | None = None
  ) -> tree.MapLikeTree[ValueT]:
    """Directly apply aggregate on inputs."""
    return self.get_result(self.update_state(self.create_state(), inputs))
