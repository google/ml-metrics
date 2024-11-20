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
import itertools as it
from typing import Any, Callable, Generic, Hashable, Iterable, Iterator, Mapping, Self, TypeVar

from ml_metrics._src.aggregates import base as aggregates
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree
from ml_metrics._src.utils import iter_utils
import more_itertools as mit
import numpy as np


SliceIteratorFn = Callable[..., Iterable[Hashable]]
SliceMaskIteratorFn = Callable[..., Iterable[tuple[tuple[Hashable, ...], Any]]]
MaskTree = tree.MapLikeTree[bool]
ValueT = TypeVar('ValueT')
FnT = TypeVar('FnT')
StateT = TypeVar('StateT')
ValueT = TypeVar('ValueT')


@dataclasses.dataclass(kw_only=True, frozen=True)
class TreeFn(Generic[FnT, ValueT], tree.MapLikeTreeCallable[ValueT]):
  """A lazy function that takes a dict and outputs either a dict or a tuple.

  Attributes:
    input_keys: The input keys used to take from the input dict. If left empty
      (default), this outputs an empty tuple.
    inpt_argkeys: The input argument names that postionally match with the
      values referenced by the input_keys, and passed in as kwargs to the fn.
    output_keys: The output keys of the function outputs. If left empty
      (default), output will be only positional (a tuple).
    fn: A callable instance that takes positional inputs and outputs positional
      value (tuple if there are multiple).
    masks: A mask or a tuple of masks that will be applied to the inputs chosen
      by `input_keys`. If one mask is provided, it will be applied to all
      inputs. If a tuple of masks is provided, each mask will be applied to the
      corresponding input in sequence of the keys in `input_keys`.
    replace_mask_false_with: If provided, replace False in the mask with this
      value.
    fn_batch_size: Overrides the input batch size of the function call.
    batch_size: Overrides the output batch size, has to be set when
      fn_batch_size is set.
    lazy: If True, the underlying function is lazy, normally, this means it
      needs to be constructed at runtime.
    id: A string that serves as an identifier for the TreeFn instance.
  """

  input_keys: tree.TreeMapKey | tree.TreeMapKeys | None = ()
  input_argkeys: tuple[str, ...] = ()
  output_keys: tree.TreeMapKey | tree.TreeMapKeys = ()
  fn: FnT | lazy_fns.LazyFn[FnT] | None = None
  masks: tuple[MaskTree, ...] = dataclasses.field(default=(), repr=False)
  replace_mask_false_with: Any = dataclasses.field(
      default=tree.DEFAULT_FILTER, repr=False
  )
  fn_batch_size: int = 0
  batch_size: int = 0
  _default_constructor: bool = dataclasses.field(default=True, repr=False)

  def __post_init__(self):
    if self._default_constructor:
      raise ValueError(
          f'Do not use the constructor, use {self.__class__.__name__}.new().'
      )
    if self.fn_batch_size and not self.batch_size:
      raise ValueError(
          'fn_batch_size should be used with batch_size, got'
          f' {self.fn_batch_size=} and {self.batch_size=}.'
      )
    if self.fn_batch_size < 0 or self.batch_size < 0:
      raise ValueError(
          'batch sizes have to be non-negative, got'
          f' {self.fn_batch_size=} and {self.batch_size=}'
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
      fn_batch_size: int = 0,
      batch_size: int = 0,
      **disable_slicing,
  ) -> TreeFn[FnT, ValueT]:
    """Normalizes the arguments before constructing a TreeFn."""
    input_argkeys = ()
    # TODO: b/311207032 - support literals in dictionary input_keys for non-data
    # arguement passing: foo(a=Key('data'), b=Literal('value'))).
    if isinstance(input_keys, Mapping):
      input_argkeys, input_keys = tuple(zip(*input_keys.items()))
    if isinstance(output_keys, Mapping):
      raise NotImplementedError(
          f'dict output_keys is not supported, got {output_keys=}.'
      )
    # These require a tuple for positional inputs. Normalize_keys converts
    # the keys into a tuple of keys to make sure the actual selected inputs
    # are also wrapped in a tuple.
    input_keys = tree.normalize_keys(input_keys)
    output_keys = tree.normalize_keys(output_keys)
    if not isinstance(masks, tuple):
      masks = (masks,)
    if not fn and input_argkeys:
      raise ValueError(
          f'Select operation should not have kw args, got {input_keys=}'
      )
    return cls(
        output_keys=output_keys,
        input_keys=input_keys,
        input_argkeys=input_argkeys,
        fn=fn,
        masks=masks,
        replace_mask_false_with=replace_mask_false_with,
        fn_batch_size=fn_batch_size,
        batch_size=batch_size,
        _default_constructor=False,
        **disable_slicing,
    )

  @functools.cached_property
  def lazy(self):
    return lazy_fns.is_resolvable(self.fn)

  @functools.cached_property
  def actual_fn(self) -> FnT:
    if self.lazy:
      return lazy_fns.maybe_make(self.fn)
    return self.fn

  @functools.cached_property
  def num_inputs(self):
    if isinstance(self.input_keys, tuple):
      return len(self.input_keys)
    return 1 if self.input_keys else 0

  @functools.cached_property
  def num_outputs(self):
    if isinstance(self.output_keys, tuple):
      return len(self.output_keys)
    return 1 if self.output_keys else 0

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

  def _normalize_outputs(
      self, outputs: tuple[ValueT, ...] | ValueT
  ) -> tuple[ValueT, ...] | tuple[tuple[ValueT, ...]]:  # pylint: disable=g-one-element-tuple
    """Normalizes the outputs to tuple of values."""
    # This is needed because function or selecting can return either a single
    # value or a tuple of values.
    if not isinstance(outputs, tuple):
      outputs = (outputs,)
    # If the output_keys is SELF, ((SELF,) after output_key normalization).
    # We again need to normalize the output due to the duality of tuple or
    # single value outputs. E.g., input_keys='a' gives (some_values, ) up to
    # this point, we need to unwrap it to some_values as the return, thus,
    # skipping the wrapping here because SELF is normalized to (SELF,).
    output_to_self = self.output_keys[0] == tree.Key.SELF
    if output_to_self and len(outputs) > 1:
      outputs = (outputs,)
    return outputs

  def get_inputs(
      self, inputs: tree.MapLikeTree[ValueT] | None
  ) -> tuple[ValueT, ...]:
    fn_inputs = tree.TreeMapView.as_view(inputs)[self.input_keys]
    # A tuple of masks will apply to each input per input_keys. Otherwise, the
    # masks will be applied to all inputs.
    if self.masks:
      fn_inputs = self._apply_masks(fn_inputs)
    return fn_inputs

  def _maybe_call_fn(self, fn_inputs: tuple[ValueT, ...]) -> Any:
    if self.actual_fn is not None:
      try:
        if self.input_argkeys:
          outputs = self.actual_fn(**dict(zip(self.input_argkeys, fn_inputs)))
        else:
          outputs = self.actual_fn(*fn_inputs)
      except Exception as e:
        raise ValueError(
            f'Failed to call {self.fn} with inputs:'
            f' {tree.tree_shape(fn_inputs)}'
        ) from e
    else:
      # If the function is None, this serves as a select operation.
      outputs = fn_inputs
    return self._normalize_outputs(outputs)

  def _get_outputs(
      self, outputs: Any, inputs: Any = tree.NullMap()
  ) -> tree.MapLikeTree[ValueT]:
    return tree.TreeMapView(inputs).copy_and_set(self.output_keys, outputs).data

  def __call__(
      self, inputs: tree.MapLikeTree[ValueT] | None = None
  ) -> tree.MapLikeTree[ValueT] | None:
    return mit.first(self.iterate([inputs]))

  def _iterate(
      self, input_iterator: Iterable[tree.MapLikeTree[ValueT] | None]
  ) -> Iterable[tree.MapLikeTree[ValueT] | None]:
    input_iterator = iter(input_iterator)
    fn_inputs = iter_utils.rebatched_tuples(
        map(self.get_inputs, input_iterator),
        batch_size=self.fn_batch_size,
        num_columns=self.num_inputs,
    )
    yield from iter_utils.rebatched_tuples(
        map(self._maybe_call_fn, fn_inputs),
        batch_size=self.batch_size,
        num_columns=self.num_outputs,
    )

  def iterate(
      self, input_iterator: Iterable[tree.MapLikeTree[ValueT] | None]
  ) -> Iterable[tree.MapLikeTree[ValueT] | None]:
    yield from map(self._get_outputs, self._iterate(input_iterator))

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

  def iterate(
      self, input_iterator: Iterable[tree.MapLikeTree[ValueT] | None]
  ) -> Iterable[tree.MapLikeTree[ValueT] | None]:
    yield from it.starmap(
        self._get_outputs,
        iter_utils.processed_with_inputs(self._iterate, input_iterator),
    )


class Select(TreeFn):
  """A lazy Map operation that operates on an mappable."""

  def __post_init__(self):
    assert self.fn is None, f'Select should have no fn, got {self.fn}.'


@dataclasses.dataclass(frozen=True)
class SliceKey:
  features: tuple[tree.TreeMapKey, ...] = ()
  values: tuple[Hashable, ...] = ()

  def __post_init__(self):
    assert isinstance(self.features, tuple)
    assert isinstance(self.values, tuple)
    if len(self.features) != len(self.values):
      raise ValueError(
          'SliceKey should have same number of features and values, got'
          f' {self.features=} and {self.values=}. It is possible slice_name is'
          ' not matching with the slice_fn output.'
      )


@dataclasses.dataclass(frozen=True)
class Slicer:
  """Generates slices given an input_keys and slicer iterator.

  Note that the slicer_fn takes one row as input instead of a batch.
  """

  slice_mask_fn: (
      SliceMaskIteratorFn | lazy_fns.LazyFn[SliceMaskIteratorFn] | None
  ) = None
  slice_name: str | tuple[str, ...] = ()
  input_keys: tree.TreeMapKey | tree.TreeMapKeys = ()
  within_values: tuple[Any, ...] = ()
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
    within_values = ()
    if isinstance(input_keys, Mapping):
      input_keys, within_values = tuple(zip(*input_keys.items()))
      within_values = tuple(map(tree.normalize_keys, within_values))
    if within_values and (slice_fn or slice_mask_fn):
      raise ValueError(
          'Cannot provide within_values when either slice_fn or slice_mask_fn'
          ' is provided. Within values should be handled directly by slice_fn.'
      )
    # Fn requires a tuple for positional inputs. Normalize_keys converts
    # the keys into a tuple of keys to make sure the actual selected inputs
    # are also wrapped in a tuple.
    input_keys = tree.normalize_keys(input_keys)
    slice_name = tree.normalize_keys(slice_name) or input_keys

    def _default_slice_fn(*args):
      if not within_values:
        return (args,)
      return (
          arg
          for arg, within_value in zip(args, within_values, strict=True)
          if arg in within_value
      )

    slice_fn = slice_fn or _default_slice_fn

    def _slice_mask_fn(*inputs):
      mask_by_slice = collections.defaultdict(list)
      batch_ix = 0
      for row in zip(*inputs):
        for slice_value in slice_fn(*row):
          # slice_name is always normalized to a tuple, so slice_values have to
          # be normalized here as well.
          if not isinstance(slice_value, tuple):
            slice_value = (slice_value,)
          try:
            mask_by_slice[slice_value].append(batch_ix)
          except TypeError as e:
            raise TypeError(
                f'{slice_value=} generated by {slice_name=} not hashable.'
            ) from e
        batch_ix += 1

      # Re-batch the masks per batch.
      batch_size = batch_ix
      for slice_value, indices in mask_by_slice.items():
        mask = np.zeros(batch_size).astype(bool)
        mask[indices] = True
        yield slice_value, (mask,)

    slice_mask_fn = slice_mask_fn or _slice_mask_fn
    return cls(
        slice_mask_fn=slice_mask_fn,
        slice_name=slice_name,
        input_keys=input_keys,
        replace_mask_false_with=replace_mask_false_with,
        within_values=within_values,
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
    for slice_value, mask in self.slice_mask_fn(*inputs):
      if not isinstance(slice_value, tuple):
        slice_value = (slice_value,)
      yield SliceKey(self.slice_name, slice_value), mask

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
  disable_slicing: bool = False

  def create_state(self) -> StateT:
    try:
      return self.actual_fn.create_state()
    except Exception as e:
      raise ValueError(f'Cannot create_state in {self=}') from e

  def update_state(
      self, state: StateT, inputs: tree.MapLikeTree[ValueT]
  ) -> StateT:
    try:
      fn_inputs = self.get_inputs(inputs)
      if self.input_argkeys:
        fn_inputs = dict(zip(self.input_argkeys, fn_inputs))
        state = self.actual_fn.update_state(state, **fn_inputs)
      else:
        state = self.actual_fn.update_state(state, *fn_inputs)
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
    return self._get_outputs(self._normalize_outputs(outputs))

  def __call__(
      self, inputs: tree.MapLikeTree[ValueT] | None = None
  ) -> tree.MapLikeTree[ValueT] | None:
    """Directly apply aggregate on inputs."""
    return self.get_result(self.update_state(self.create_state(), inputs))

  # TODO: b/356633410 - support iterate for TreeAggregateFn.
  def iterate(
      self, input_iterator: Iterable[tree.MapLikeTree | None]
  ) -> Iterable[tree.MapLikeTree | None]:
    raise NotImplementedError(
        f'TreeAggregateFn does not support iterate, TreeAggFn:{self}'
    )
