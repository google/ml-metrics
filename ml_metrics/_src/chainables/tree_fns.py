# Copyright 2025 Google LLC
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
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
import dataclasses as dc
import functools
import itertools as itt
from typing import Any, Generic, Self, TypeVar

from ml_metrics._src import types
from ml_metrics._src.aggregates import base as aggregates
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree
from ml_metrics._src.utils import iter_utils
import more_itertools as mit
import numpy as np


_Aggregatable = aggregates.Aggregatable | aggregates.HasAsAggFn
SliceIteratorFn = Callable[..., Iterable[Hashable]]
SliceMaskIteratorFn = Callable[..., Iterable[tuple[tuple[Hashable, ...], Any]]]
MaskTree = tree.TreeLike[bool]
_T = TypeVar('_T')
_FnT = TypeVar('_FnT')
StateT = TypeVar('StateT')


def _identity_fn(*x):
  return x


@dc.dataclass(kw_only=True, frozen=True)
class TreeFn(Generic[_FnT, _T]):
  """A lazy function that takes a dict and outputs either a dict or a tuple.

  Attributes:
    input_keys: The input keys used to take from the input dict. If left empty
      (default), this outputs an empty tuple.
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
    input_batch_size: Tracker for the input batch size, unbatched inputs when
      this is 0 (default).
    batch_size: Rebatch the output to `batch_size`.
    output_batch_size: The batch size of the output with consideration of the
      `input_batch_size` and `batch_size`.
    lazy: If True, the underlying function is lazy, normally, this means it
      needs to be constructed at runtime.
    id: A string that serves as an identifier for the TreeFn instance.
    ignore_error: If True, ignore the error when calling the function.
  """

  input_keys: tree.TreeMapKey | tree.TreeMapKeys = (tree.Key.SELF,)
  output_keys: tree.TreeMapKey | tree.TreeMapKeys = (tree.Key.SELF,)
  fn: _FnT | lazy_fns.LazyFn[_FnT] | None = None
  masks: tuple[MaskTree, ...] | MaskTree = dc.field(default=(), repr=False)
  replace_mask_false_with: Any = dc.field(
      default=tree.DEFAULT_FILTER, repr=False
  )
  input_batch_size: int = 0
  batch_size: int = 0
  ignore_error: bool = False
  _input_keys: tuple[tree.TreeMapKey, ...] = ()
  _input_argkeys: tuple[str, ...] = ()
  _cached_fn: _FnT | None = None

  def __post_init__(self):
    if self.batch_size < 0:
      raise ValueError(
          f'batch sizes have to be non-negative, got {self.batch_size=}'
      )
    input_argkeys = ()
    input_keys, output_keys = self.input_keys, self.output_keys
    if isinstance(input_keys, Mapping):
      input_argkeys, input_keys = tuple(zip(*input_keys.items()))
    # These require a tuple for positional inputs. Normalize_keys converts
    # the keys into a tuple of keys to make sure the actual selected inputs
    # are also wrapped in a tuple.
    object.__setattr__(self, '_input_keys', tree.normalize_keys(input_keys))
    object.__setattr__(self, '_input_argkeys', input_argkeys)
    object.__setattr__(self, 'output_keys', tree.normalize_keys(output_keys))
    if not isinstance(self.masks, tuple):
      object.__setattr__(self, 'masks', (self.masks,))
    if self.fn is None:
      if input_argkeys:
        raise ValueError(f'Select Op cannot have kwargs, got {input_keys=}')
      object.__setattr__(self, 'fn', _identity_fn)

  @functools.cached_property
  def _lazy(self):
    fn = self.fn
    return types.is_resolvable(fn)

  @property
  def _actual_fn(self) -> _FnT:
    """Returns the actual function."""
    if self._cached_fn is not None:
      return self._cached_fn

    result = lazy_fns.maybe_make(self.fn) if self._lazy else self.fn
    assert not types.is_resolvable(result), f'{type(result)=}'
    object.__setattr__(self, '_cached_fn', result)
    return result

  @functools.cached_property
  def _num_inputs(self):
    """Returns the number of inputs."""
    if isinstance(self._input_keys, tuple):
      return len(self._input_keys)
    return 1 if self._input_keys else 0

  @functools.cached_property
  def _num_outputs(self):
    """Returns the number of outputs."""
    if isinstance(self.output_keys, tuple):
      return len(self.output_keys)
    return 1 if self.output_keys else 0

  @property
  def output_batch_size(self) -> int:
    """Returns the output batch size."""
    return self.batch_size or self.input_batch_size

  def maybe_make(self: Self) -> Self:
    """Explicitly instantiate the lazy_fn of a tree_fn."""
    return dc.replace(self, fn=lazy_fns.maybe_make(self.fn))

  def with_masks(
      self,
      masks: tuple[MaskTree, ...],
      replace_mask_false_with: Any = tree.DEFAULT_FILTER,
  ) -> Self:
    """Returns a new TreeFn with the masks."""
    return dc.replace(
        self, masks=masks, replace_mask_false_with=replace_mask_false_with
    )

  def _apply_masks(self, items: tuple[tree.TreeLike, ...]):
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
      self, outputs: tuple[_T, ...] | _T
  ) -> tuple[_T, ...] | tuple[tuple[_T, ...]]:  # pylint: disable=g-one-element-tuple
    """Normalizes the outputs to tuple of values."""
    # This is needed because function or selecting can return either a single
    # value or a tuple of values.
    if not isinstance(outputs, tuple):
      outputs = (outputs,)
    # If the output_keys is SELF, ((SELF,) or () after output_key normalization)
    # We again need to normalize the output due to the duality of tuple or
    # single value outputs. E.g., input_keys='a' gives a single value output
    # (some_values,) up to this point, we need to unwrap it to some_values as
    # the return, thus, skipping the wrapping here because SELF is normalized
    # to (SELF,).
    output_to_self = not self.output_keys or self.output_keys == (
        tree.Key.SELF,
    )
    if output_to_self and len(outputs) > 1:
      outputs = (outputs,)
    return outputs

  def _get_inputs(self, inputs: tree.TreeLike) -> tuple[_T, ...]:
    try:
      fn_inputs = tree.TreeMapView.as_view(inputs)[self._input_keys]
      # A tuple of masks will apply to each input per input_keys. Otherwise, the
      # masks will be applied to all inputs.
      if self.masks:
        fn_inputs = self._apply_masks(fn_inputs)
    except Exception as e:
      raise KeyError(
          f'Failed to get inputs {self.input_keys=} in {self}'
      ) from e
    return fn_inputs

  def _maybe_call_fn(self, fn_inputs: tuple[_T, ...]):
    """Calls the function with the inputs."""
    try:
      kw_inputs = {}
      if self._input_argkeys:
        fn_inputs, kw_inputs = (), dict(zip(self._input_argkeys, fn_inputs))
        # If the function is None, this serves as a select operation.
        assert self._actual_fn is not _identity_fn, f'{kw_inputs=}'
      return self._actual_fn(*fn_inputs, **kw_inputs)
    except Exception as e:
      keys = [tree.Key().at(i) for i in range(len(fn_inputs))]
      shape = tree.tree_shape(fn_inputs, keys)
      raise ValueError(f'Failed to call {self.fn} with inputs {shape=}') from e

  def _get_outputs(
      self, outputs: Any, inputs: Any = tree.NullMap()
  ) -> tree.TreeLike[_T]:
    """Returns the outputs with the input keys."""
    result = tree.TreeMapView(inputs)
    if len(self.output_keys) == 1 and len(outputs) > 1:
      return result.copy_and_set(self.output_keys, outputs).data
    for keys, output in zip(self.output_keys, outputs, strict=True):
      if isinstance(keys, Mapping):
        values = tree.TreeMapView(output)[tuple(keys.values())]
        result = result.copy_and_set(tuple(keys.keys()), values)
      else:
        result = result.copy_and_set(keys, output)
    return result.data

  def __call__(self, inputs: tree.TreeLike = None) -> tree.TreeLike[_T]:
    return next(self.iterate([inputs]))

  def _iterator_fn(
      self, fn_inputs: Iterator[tuple[Any, ...]], ignore_error: bool = False
  ) -> Iterator[Any]:
    """Returns the iterator function."""
    # Only ignore function call error.
    map_ = iter_utils.map_ignore_error if ignore_error else map
    return map_(self._maybe_call_fn, fn_inputs)

  def _iterate(
      self, input_iterator: Iterator[tree.TreeLike], ignore_error: bool = False
  ) -> Iterator[tree.TreeLike[_T]]:
    """Iterates through the input_iterator and calls the function."""
    fn_inputs = map(self._get_inputs, input_iterator)
    fn_outputs = self._iterator_fn(fn_inputs, ignore_error)
    fn_outputs = map(self._normalize_outputs, fn_outputs)
    if self.batch_size:
      fn_outputs = iter_utils.rebatched_args(
          fn_outputs,
          batch_size=self.batch_size,
          num_columns=self._num_outputs,
      )
    return fn_outputs

  def iterate(
      self, input_iterator: Iterable[tree.TreeLike]
  ) -> Iterator[tree.TreeLike[_T]]:
    return map(
        self._get_outputs,
        self._iterate(iter(input_iterator), ignore_error=self.ignore_error),
    )

  def __getstate__(self):
    state = self.__dict__.copy()
    state.pop('_lazy', None)
    state.pop('_num_outputs', None)
    state.pop('_num_inputs', None)
    state.pop('_actual_fn', None)
    state.pop('_cached_fn', None)
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)


def _flatten_tuple(*columns: tuple[Any, ...]) -> Any:
  """Flattens a tuple of values."""
  for row in zip(*columns, strict=True):
    if len(row) == 1:
      yield row[0]
    else:
      yield row


class FlattenFn(TreeFn):
  """A lazy Flatten operation that operates on an mappable."""

  def __post_init__(self):
    super().__post_init__()
    if self.fn is _identity_fn:
      object.__setattr__(self, 'fn', _flatten_tuple)

  def _iterator_fn(
      self, fn_inputs: Iterator[tuple[Any, ...]], ignore_error: bool = False
  ) -> Iterator[Any]:
    """Returns the iterator function."""
    # Only ignore function call error.
    map_ = iter_utils.map_ignore_error if ignore_error else map
    outputs = map_(self._maybe_call_fn, fn_inputs)
    it = mit.peekable(outputs)
    if not isinstance(it.peek(), Iterable):
      raise TypeError(f'fn should be an iterable, got {it.peek()=}.')
    return itt.chain.from_iterable(it)


class FilterFn(TreeFn):
  """A lazy Map operation that operates on an mappable."""

  def __post_init__(self):
    super().__post_init__()
    assert not self.batch_size, f'{self.batch_size=}'

  def iterate(
      self, input_iterator: Iterable[tree.TreeLike]
  ) -> Iterator[tree.TreeLike[_T]]:
    it_ = iter_utils.processed_with_inputs(self._iterate, iter(input_iterator))
    return (elem for (value,), elem in it_ if value)


class Assign(TreeFn):
  """A lazy Map operation that operates on an mappable."""

  def __post_init__(self):
    super().__post_init__()
    assert self.output_keys, f'{self.output_keys=}'
    assert not self.batch_size, f'{self.batch_size=}'

  def iterate(
      self, input_iterator: Iterable[tree.TreeLike]
  ) -> Iterator[tree.TreeLike[_T]]:
    return itt.starmap(
        self._get_outputs,
        iter_utils.processed_with_inputs(
            self._iterate, iter(input_iterator), ignore_error=self.ignore_error
        ),
    )


class Select(TreeFn):
  """A lazy Map operation that operates on an mappable."""

  def __post_init__(self):
    super().__post_init__()
    assert self.fn is _identity_fn, f'Select should have no fn, got {self.fn}.'
    assert not self.batch_size, f'{self.batch_size=}'


class _CallableSink(Callable[..., None]):
  """A sink function that can be used to sink the data."""

  def __init__(self, sink: types.SinkT):
    self._sink = sink

  def __call__(self, *data: tree.TreeLike[_T]) -> None:
    """Writes the data to the sink."""
    self._sink.write(*data)

  def close(self) -> None:
    """Closes the sink."""
    self._sink.close()


class Sink(TreeFn[types.SinkT, _T]):
  """A TreeFn that sink and forward the inputs."""

  def __post_init__(self):
    super().__post_init__()
    if not isinstance(super()._actual_fn, types.SinkT):
      raise TypeError(f'The fn is not a sink, got {type(self._actual_fn)=}')

  @property
  def _actual_fn(self) -> _CallableSink:
    """Returns the actual function."""
    return _CallableSink(super()._actual_fn)

  def iterate(
      self, input_iterator: Iterable[tree.TreeLike]
  ) -> Iterator[tree.TreeLike[_T]]:
    try:
      it_ = iter_utils.processed_with_inputs(
          self._iterate, iter(input_iterator), ignore_error=self.ignore_error
      )
      yield from (elem for _, elem in it_)
    finally:
      self._actual_fn.close()


@dc.dataclass(frozen=True)
class SliceKey:
  """A key that represents a slice."""

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

  def __len__(self) -> int:
    assert len(self.features) == len(self.values), f'got {self}'
    return len(self.features)


@dc.dataclass(frozen=True)
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
  _default_constructor: bool = dc.field(default=True, repr=False)

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
        slice_name=slice_name,  # pytype: disable=wrong-arg-types
        input_keys=input_keys,
        replace_mask_false_with=replace_mask_false_with,
        within_values=within_values,
        _default_constructor=False,
    )

  def iterate_and_slice(
      self, inputs: tree.TreeLike
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
    return dc.replace(
        self, slice_mask_fn=lazy_fns.maybe_make(self.slice_mask_fn)
    )


@dc.dataclass(kw_only=True, frozen=True)
class TreeAggregateFn(Generic[_T, StateT], TreeFn[_Aggregatable, _T]):
  """Transform with one AggregateFn with its input and output keys."""

  disable_slicing: bool = False

  def __post_init__(self):
    super().__post_init__()
    # Check whether the actual_fn is an Aggregatable.
    _ = self._actual_fn

  @property
  def _actual_fn(self) -> aggregates.Aggregatable:
    if (actual_fn := self._cached_fn) is not None:
      assert isinstance(actual_fn, aggregates.Aggregatable)
      return actual_fn

    actual_fn = lazy_fns.maybe_make(self.fn) if self._lazy else self.fn
    if aggregates.has_as_agg_fn(actual_fn):
      actual_fn = actual_fn.as_agg_fn()
    if not isinstance(actual_fn, aggregates.Aggregatable):
      raise TypeError(
          'Not an aggregatable, either has to be an istance of Aggregatable or'
          ' has to have `as_agg_fn` method to convert it to an Aggregatable,'
          f' got a {type(actual_fn)}, {actual_fn=}'
      )
    object.__setattr__(self, '_cached_fn', actual_fn)
    return actual_fn

  def create_state(self) -> StateT:
    try:
      return self._actual_fn.create_state()
    except Exception as e:
      raise ValueError(f'Cannot create_state in {self=}') from e

  def update_state(self, state: StateT, inputs: tree.TreeLike) -> StateT:
    """Updates the state by inputs."""
    try:
      fn_inputs, kw_inputs = self._get_inputs(inputs), {}
      if self._input_argkeys:
        fn_inputs, kw_inputs = (), dict(zip(self._input_argkeys, fn_inputs))
      state = self._actual_fn.update_state(state, *fn_inputs, **kw_inputs)
    except Exception as e:
      input_keys = self._input_keys
      key_paths = tuple(self._input_keys)
      if self._input_argkeys:
        input_keys = dict(zip(self._input_argkeys, self._input_keys))
      raise ValueError(
          f'Cannot call with {input_keys=}, {self.output_keys=},'
          f' {type(self.fn)} with'
          f' shape:\n{tree.tree_shape(inputs, key_paths=key_paths)}'
      ) from e
    return state

  def merge_states(self, states: StateT) -> StateT:
    return self._actual_fn.merge_states(states)

  def get_result(self, state: StateT) -> tree.TreeLike:
    outputs = self._actual_fn.get_result(state)
    return self._get_outputs(self._normalize_outputs(outputs))

  def __call__(self, inputs: tree.TreeLike = None) -> tree.TreeLike:
    """Directly apply aggregate on inputs."""
    return self.get_result(self.update_state(self.create_state(), inputs))

  # TODO: b/356633410 - support iterate for TreeAggregateFn.
  def iterate(
      self, input_iterator: Iterable[tree.TreeLike]
  ) -> Iterator[tree.TreeLike[_T]]:
    raise NotImplementedError(
        f'TreeAggregateFn does not support iterate, TreeAggFn:{self}'
    )
