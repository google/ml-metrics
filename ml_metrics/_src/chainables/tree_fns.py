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

import collections
import dataclasses
import functools
import typing
from typing import Any, Callable, Generic, Hashable, Iterable, Iterator, Mapping, Self, TypeVar

from ml_metrics._src.aggregates import base as aggregates
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree

SliceIteratorFn = Callable[..., Iterable[Hashable]]
SliceMaskIteratorFn = Callable[..., Iterable[tuple[Hashable, Any]]]
MaskTree = tree.MapLikeTree[bool]
ValueT = TypeVar('ValueT')
FnT = TypeVar('FnT')
StateT = TypeVar('StateT')
ValueT = TypeVar('ValueT')


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
    masks: A mask or a tuple of masks that will be applied to the inputs chosen
      by `input_keys`. If one mask is provided, it will be applied to all
      inputs. If a tuple of masks is provided, each mask will be applied to the
      corresponding input in sequence of the keys in `input_keys`.
    replace_mask_false_with: If provided, replace False in the mask with this
      value.
    lazy: If True, the underlying function is lazy, normally, this means it
      needs to be constructed at runtime.
    id: A string that serves as an identifier for the TreeFn instance.
  """

  input_keys: tree.TreeMapKey | tree.TreeMapKeys | None = ()
  output_keys: tree.TreeMapKey | tree.TreeMapKeys = ()
  fn: FnT | lazy_fns.LazyFn[FnT] | None = None
  masks: tuple[MaskTree, ...] = ()
  replace_mask_false_with: Any = tree.DEFAULT_FILTER
  _default_constructor: bool = dataclasses.field(default=True, repr=False)

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
      masks: tuple[MaskTree, ...] | MaskTree = (),
      replace_mask_false_with: Any = tree.DEFAULT_FILTER,
  ) -> TreeFn[FnT, ValueT]:
    """Normalizes the arguments before constructing a TreeFn."""
    if masks or fn or output_keys is not tree.Key.SELF:
      # These require a tuple for positional inputs. Normalize_keys converts
      # the keys into a tuple of keys to make sure the actual selected inputs
      # are also wrapped in a tuple.
      input_keys = tree.normalize_keys(input_keys)
    if not isinstance(masks, tuple):
      masks = (masks,)
    if output_keys is not tree.Key.SELF:
      output_keys = tree.normalize_keys(output_keys)
    return cls(
        output_keys=output_keys,
        input_keys=input_keys,
        fn=fn,
        masks=masks,
        replace_mask_false_with=replace_mask_false_with,
        _default_constructor=False,
    )

  @property
  def lazy(self):
    return lazy_fns.makeables[type(self.fn)] is not None

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

  def with_masks(
      self,
      masks: tuple[MaskTree, ...],
      replace_mask_false_with: Any = tree.DEFAULT_FILTER,
  ) -> Self:
    """Returns a new TreeFn with the masks."""
    return dataclasses.replace(
        self, masks=masks, replace_mask_false_with=replace_mask_false_with
    )

  def _apply_masks(self, items: tuple[tree.MapLikeTree[Any], ...]):
    """Applies multiple masks to multiple inputs."""
    result = []
    apply_mask_fn = functools.partial(
        tree.apply_mask,
        replace_false_with=self.replace_mask_false_with,
    )
    match self.masks:
      case (mask,):
        for item in items:
          result.append(apply_mask_fn(item, masks=mask))
      case (_, *_):
        for item, mask in zip(items, self.masks, strict=True):
          result.append(apply_mask_fn(item, masks=mask))
    return tuple(result)

  def get_inputs(
      self, inputs: tree.MapLikeTree[ValueT] | None
  ) -> tuple[ValueT, ...] | ValueT | dict[str, ValueT] | None:
    argkeys = None
    if isinstance(self.input_keys, Mapping):
      argkeys, input_keys = tuple(zip(*self.input_keys.items()))
    else:
      input_keys = self.input_keys
    fn_inputs = tree.TreeMapView.as_view(inputs)[input_keys]
    # A tuple of masks will apply to each input per input_keys. Otherwise, the
    # masks will be applied to all inputs.
    if self.masks:
      assert isinstance(fn_inputs, tuple)
      fn_inputs = self._apply_masks(fn_inputs)
    if argkeys:
      return (), dict(zip(argkeys, fn_inputs))
    else:
      return fn_inputs, {}

  def __call__(
      self, inputs: tree.MapLikeTree[ValueT] | None = None
  ) -> tree.MapLikeTree[ValueT] | None:
    fn_inputs, fn_kw_inputs = self.get_inputs(inputs)
    if (fn := self.actual_fn) is not None:
      try:
        outputs = fn(*fn_inputs, **fn_kw_inputs)
      except Exception as e:
        raise ValueError(
            f'Failed to call {self.fn} with inputs:'
            f' {tree.tree_shape(fn_inputs)}'
        ) from e
    else:
      # If the function is None, this serves as a select operation.
      if fn_kw_inputs:
        raise ValueError(
            f'Select operation should not have kw_inputs, got {fn_kw_inputs=}'
        )
      outputs = fn_inputs
    # Uses TreeMapView to handle multiple key.
    return tree.TreeMapView().copy_and_set(self.output_keys, outputs).data

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
  ) -> tree.MapLikeTree | None:
    fn_inputs, kwargs_fn_inputs = self.get_inputs(inputs)
    if (fn := self.actual_fn) is not None:
      try:
        outputs = fn(*fn_inputs, **kwargs_fn_inputs)
      except Exception as e:
        raise ValueError(f'Failed to call {self.fn} with {fn_inputs=}') from e
    else:
      # If the function is None, this serves as a select operation.
      outputs = fn_inputs
    # Uses TreeMapView to handle multiple key.
    return tree.TreeMapView(inputs).copy_and_set(self.output_keys, outputs).data


class Select(TreeFn):
  """A lazy Map operation that operates on an mappable."""

  def __post_init__(self):
    assert self.fn is None, f'Select should have no fn, got {self.fn}.'


@dataclasses.dataclass(frozen=True)
class SliceKey:
  features: tuple[tree.TreeMapKey, ...] = ()
  values: tuple[Any, ...] = ()

  def __post_init__(self):
    assert isinstance(self.features, tuple)
    assert isinstance(self.values, tuple)
    if len(self.features) != len(self.values):
      raise ValueError(
          'SliceKey should have same number of features and values, got'
          f' {self.features=} and {self.values=}'
      )


@dataclasses.dataclass(frozen=True)
class Slicer:
  """Generates slices given an input_keys and slicer iterator.

  Note that the slicer_fn takes one row as input instead of a batch.
  """

  slice_mask_fn: (
      SliceMaskIteratorFn | lazy_fns.LazyFn[SliceMaskIteratorFn] | None
  ) = None
  slice_input_keys: tree.TreeMapKey | tree.TreeMapKeys = ()
  slice_name: str | tuple[str, ...] = ()
  input_keys: tree.TreeMapKey | tree.TreeMapKeys = ()
  replace_mask_false_with: Any = tree.DEFAULT_FILTER
  _default_constructor: bool = dataclasses.field(default=True, repr=False)

  def __post_init__(self):
    if self._default_constructor:
      raise ValueError(
          f'Do not use the constructor, use {self.__class__.__name__}.new().'
      )

  @classmethod
  def new(
      cls,
      input_keys: tree.TreeMapKey | tree.TreeMapKeys = tree.Key.SELF,
      *,
      slice_name: tree.TreeMapKey | tree.TreeMapKeys = tree.Key.SELF,
      slice_fn: SliceIteratorFn | None = None,
      slice_mask_fn: SliceMaskIteratorFn | None = None,
      replace_mask_false_with: Any = tree.DEFAULT_FILTER,
  ) -> Self:
    """Normalizes the arguments before constructing a TreeFn."""
    if slice_fn and slice_mask_fn:
      raise ValueError(
          'Cannot provide both slice_fn and slice_mask_fn, got'
          f' {slice_fn=} and {slice_mask_fn=}.'
      )
    # Default slice_fns directly takes the slice_inputs.
    slice_fn = slice_fn or (lambda *args: (args,))

    # Converts a pure slice_fn to a slice_mask_fn with dummy masks.
    def _slice_mask_fn(*args):
      for slice_value in slice_fn(*args):
        yield slice_value, (True,)

    slice_mask_fn = slice_mask_fn or _slice_mask_fn
    if slice_mask_fn and not slice_name:
      raise ValueError(
          'Must provide slice_name when either slice_fn or slice_mask_fn is'
          ' provided.'
      )
    # Fn requires a tuple for positional inputs. Normalize_keys converts
    # the keys into a tuple of keys to make sure the actual selected inputs
    # are also wrapped in a tuple.
    input_keys = tree.normalize_keys(input_keys)
    slice_name = tree.normalize_keys(slice_name) or input_keys
    return cls(
        slice_mask_fn=slice_mask_fn,
        slice_name=slice_name,
        input_keys=input_keys,
        replace_mask_false_with=replace_mask_false_with,
        _default_constructor=False,
    )

  def iterate_and_slice(
      self, inputs: tree.MapLikeTree[Any]
  ) -> Iterator[tuple[SliceKey, Any]]:
    """Iterates through each row of the batch and emits slice and row index."""
    # The input_keys are always normalized to tuple, so the output will always
    # be a tuple.
    inputs = tuple(
        tree.TreeMapView.as_view(inputs, key_paths=self.input_keys).values()
    )
    distinct_values = collections.defaultdict(list)
    batch_ix = 0
    masks = ()
    for row in zip(*inputs):
      for slice_value, masks in self.slice_mask_fn(*row):
        # slice_name is always normalized to a tuple, so slice_values have to
        # be normalized here as well.
        if not isinstance(slice_value, tuple):
          slice_value = (slice_value,)
        slice_key = SliceKey(self.slice_name, slice_value)
        try:
          distinct_values[slice_key].append((batch_ix, masks))
        except TypeError as e:
          raise TypeError(
              f'{slice_key=} generated by {self.slice_name=} not hashable.'
          ) from e
      batch_ix += 1

    # Re-batch the masks per batch.
    mask_size = len(masks)
    for slice_key, ix_masks_pair in distinct_values.items():
      batch_mask = [(False,) * mask_size] * batch_ix
      assigned_row = set()
      for ix, masks in ix_masks_pair:
        if ix in assigned_row:
          raise ValueError(
              f'Duplicate index {ix=} for {slice_key=} with mask'
              f' {batch_mask[ix]}'
          )
        assigned_row.add(ix)
        batch_mask[ix] = masks if isinstance(masks, tuple) else (masks,)
      # transpose each mask per column.
      batch_masks = tuple(zip(*batch_mask))
      yield slice_key, batch_masks

  def maybe_make(self) -> Self:
    return dataclasses.replace(
        self, slice_mask_fn=lazy_fns.maybe_make(self.slice_mask_fn)
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class TreeAggregateFn(
    Generic[StateT, ValueT], TreeFn[aggregates.Aggregatable, ValueT]
):
  """Transform with one AggregateFn with its input and output keys."""

  fn: (
      aggregates.Aggregatable | lazy_fns.LazyFn[aggregates.Aggregatable] | None
  ) = None

  def create_state(self) -> StateT:
    try:
      return self.actual_fn.create_state()
    except Exception as e:
      raise ValueError(f'Cannot create_state in {self=}') from e

  def update_state(
      self, state: StateT, inputs: tree.MapLikeTree[ValueT]
  ) -> StateT:
    try:
      arg_inputs, kwargs_inputs = self.get_inputs(inputs)
      state = self.actual_fn.update_state(state, *arg_inputs, **kwargs_inputs)
    except Exception as e:
      raise ValueError(
          f'Cannot call {self.input_keys=}, {self.output_keys=},'
          f' {type(self.fn)} with shape:\n{tree.tree_shape(inputs)}'
      ) from e
    return state

  def merge_states(self, states: StateT) -> StateT:
    return self.actual_fn.merge_states(states)

  def get_result(self, state: StateT) -> tree.MapLikeTree[ValueT] | None:
    outputs = self.actual_fn.get_result(state)
    return tree.TreeMapView().copy_and_set(self.output_keys, outputs).data

  def __call__(
      self, inputs: tree.MapLikeTree[ValueT] | None = None
  ) -> tree.MapLikeTree[ValueT] | None:
    """Directly apply aggregate on inputs."""
    return self.get_result(self.update_state(self.create_state(), inputs))
