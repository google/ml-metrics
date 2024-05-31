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
"""Library for the Python nested MapLikes (Sequence/Mapping)."""

from __future__ import annotations

import abc
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
import copy
import dataclasses
from typing import Any, Protocol, TypeVar, Union

from ml_metrics._src import base_types


def tree_shape(inputs: Any):
  result = {}
  try:
    for k, v in TreeMapView.as_view(inputs).items():
      if hasattr(v, 'shape'):
        result[k] = v.shape
      else:
        result[k] = type(v)
  except Exception:  # pylint: disable=broad-exception-caught
    result = inputs
  return result


class Index(int):
  """Convenient class to identify sequence index."""

  def __repr__(self):
    return f'Index({super().__repr__()})'


class Reserved(str):
  """Convenient type to indicate to select all in the Tree."""

  def __repr__(self):
    return f'Reserved({super().__repr__()})'

  def __iter__(self) -> Iterator[str]:
    # Avoids iterate over the str itself.
    yield self


BaseKey = Hashable | Index
LeafValueT = TypeVar('LeafValueT')

KT = TypeVar('KT', bound=BaseKey)
VT = TypeVar('VT')


_SKIP = Reserved('SKIP')
_SELF = Reserved('SELF')


class MapLike(Protocol[KT, VT]):
  """An interface that implements data[x] covering `Mapping` and `Sequence`."""

  @abc.abstractmethod
  def __getitem__(self, key: KT) -> VT:
    """Returns value for a key, returns a tuple for multi-key."""


# A MapLike tree is a nested MapLike structure. E.g., Sequence and Mapping
# DataFrame, Numpy array are all MapLike.
# Uses Union here since forward reference does not work with |.
MapLikeTree = MapLike[BaseKey, Union['MapLikeTree', LeafValueT]]
MapLikeTreeCallable = Callable[[MapLikeTree | None], MapLikeTree | None]


class MaskBehavior(base_types.StrEnum):
  """Enum for the behavior of applying masks."""

  FILTER = 'filter'
  REPLACE = 'replace'


def apply_mask(
    items: MapLikeTree[Any],
    *,
    masks: MapLikeTree[bool],
    mask_behavior: MaskBehavior = MaskBehavior.FILTER,
    replace_false_with: Any = None,
):
  """Applies masks to inputs.

  The masks and the inputs have to have the tree structure and shape, i.e.,
  there has to be a boolean per tree of elements or element in the inputs.

  Args:
    items: A tree of inputs.
    masks: A tree of masks.
    mask_behavior: The behavior of applying masks: if "filter", throw away the
      elements that are False in the masks. If "replace", replace the elements
      that are False in the masks with the replace_false_with.
    replace_false_with: The value to replace False with when replace_false is
      True.

  Returns:
    A tree of inputs with the masks applied.
  """
  if isinstance(masks, dict) and isinstance(items, dict):
    result = {}
    for key, mask in masks.items():
      if isinstance(mask, bool):
        if mask:
          result[key] = items[key]
        elif mask_behavior == MaskBehavior.REPLACE:
          result[key] = replace_false_with
      else:
        result[key] = apply_mask(
            items[key],
            masks=mask,
            mask_behavior=mask_behavior,
            replace_false_with=replace_false_with,
        )
  elif isinstance(masks, dict) or isinstance(items, dict):
    raise ValueError(
        f'Masks and inputs have to be both dict: {type(masks)=}, {type(items)=}'
    )
  else:
    result = []
    for elem, mask in zip(items, masks, strict=True):
      if isinstance(mask, bool):
        if mask:
          result.append(elem)
        elif mask_behavior == MaskBehavior.REPLACE:
          result.append(replace_false_with)
      else:
        result.append(
            apply_mask(
                elem,
                masks=mask,
                mask_behavior=mask_behavior,
                replace_false_with=replace_false_with,
            )
        )
  return result


def apply_masks(
    items: tuple[MapLikeTree[Any], ...],
    *,
    masks: tuple[MapLikeTree[bool], ...],
    mask_behavior: MaskBehavior = MaskBehavior.FILTER,
    replace_false_with: Any = None,
):
  """Applies multiple masks to multiple inputs."""
  result = []
  for item, mask in zip(items, masks, strict=True):
    result.append(
        apply_mask(
            item,
            masks=mask,
            mask_behavior=mask_behavior,
            replace_false_with=replace_false_with,
        )
    )
  return tuple(result)


def _is_key(key, other_key):
  return isinstance(key, type(other_key)) and key == other_key


class Key(tuple[BaseKey, ...]):
  """Convenient subtype of a tuple for a key path.

  This is immutable, and can only be called by either `new('a', 'b')`
  method or inputs a tuple: Path(('a', 'b')). For convenience, user can also
  attach individual keys by directly referncing the string name. E.g.,
  Path().pred.logits is same as Path('pred', 'logits) and
  Path.new('pred').logits.
  """

  # SKIP serves as a placeholder that indicates the corresponding output will be
  # ignored. This is typically only useful when there are more outputs than
  # those selected during set. Using this in get will raise a ValueError.
  @classmethod
  @property
  def SKIP(cls):  # pylint: disable=invalid-name
    return _SKIP

  # SELF means selecting the whole tree, it is equivalent to Path().
  @classmethod
  @property
  def SELF(cls):  # pylint: disable=invalid-name
    return _SELF

  @classmethod
  def Index(cls, i: int):  # pylint: disable=invalid-name
    return Index(i)

  @classmethod
  def new(cls, *args: tuple[BaseKey, ...]) -> 'Key':
    return cls(tuple(args))

  def __getattr__(self, name: str):
    return Key(self + (name,))

  def at(self, key: BaseKey):
    return Key(self + (key,))

  def __repr__(self):
    return f'Path({super().__repr__()})'


TreeMapKey = BaseKey | Key | Reserved
# For keys, if there is tuple, the first dimension is always keys dimension.
TreeMapKeys = tuple[TreeMapKey, ...]


def normalize_keys(keys: TreeMapKey | TreeMapKeys) -> TreeMapKeys:
  if isinstance(keys, Key) or not isinstance(keys, (tuple, list)):
    keys = (keys,)
  return tuple(keys)


def _default_tree(key_path: Key, value: Any):
  """Constructs a TreeLike instance given a key_path and value."""
  match key_path:
    case ():
      return value
    case (Index(key), *rest_keys):
      if key == 0:
        return [_default_tree(Key(rest_keys), value)]
      else:
        raise ValueError(
            f'Cannot insert non-zero index to empty sequence {key_path}'
        )
    case (key, *rest_keys) if isinstance(key, BaseKey):
      return {key: _default_tree(Key(rest_keys), value)}
    case _:
      raise ValueError(f'Unsupported key {key_path}')


@dataclasses.dataclass
class TreeMapView(Mapping[TreeMapKey, LeafValueT]):
  """Handler for a MapTreeLike instance as a mapping instance.

  This creates an immutable View of a `MapLikeTree`, the view implements the
  Mapping interfaces so it can be used with any TreeMapKey. Following are some
  examples:
  Given a tree = {'a': [1, 2], 'b': [3, 4]},
    tree['a'] is [1, 2]
    tree['a', 'b'] is ([1, 2], [3, 4]) (multi-key index)
  Given a nested tree = {'a': {'a1': [1, 2]}, 'b': [3, 4]}
    tree[Path('a', 'a1')] is [1, 2]
    tree[Path('a', 'a1'), 'b'] is ([1, 2], [3, 4])
  Given a tree = [{'a': [1, 2]}, {'b': [3, 4]}]
    tree[Path(Index(0), 'a')] is [1, 2]

  Attributes:
    data: Nested MapLike (Tree) structure.
    map_fn: A function that will be applied to the leaf value when accessing the
      tree.
    strict: If True, non-existing key will cause a KeyError.
    consistent_type: If True, tries to maintain a consistent container type when
      the input container is immutable. e.g., tuple will be reconstructed as
      tuple instead of list. Otherwise, the default container type (list for
      immutable Sequence)
  """

  data: MapLikeTree = ()
  map_fn: Callable[..., Any] | None = dataclasses.field(
      kw_only=True,
      default=None,
  )
  strict: bool = dataclasses.field(kw_only=True, default=False)
  consistent_type: bool = dataclasses.field(kw_only=True, default=False)

  @classmethod
  def as_view(
      cls,
      tree_or_view: MapLikeTree[LeafValueT] | TreeMapView[LeafValueT],
      map_fn: Callable[..., Any] | None = None,
  ) -> TreeMapView:
    """Util to use a MapLikeTree as a Map."""
    if not isinstance(tree_or_view, TreeMapView):
      tree_or_view = TreeMapView(tree_or_view)
    if map_fn:
      return dataclasses.replace(tree_or_view, map_fn=map_fn)
    return tree_or_view

  def _maybe_map(self, data):
    return self.map_fn(data) if self.map_fn else data

  def __get(self, key: TreeMapKey) -> MapLikeTree[LeafValueT] | None:
    """Gets the value from a single Key or Path."""
    key = key if isinstance(key, Key) else Key((key,))
    data = self.data
    for k in key:
      if _is_key(k, Key.SELF):
        return self._maybe_map(data)
      if isinstance(k, Index) and isinstance(data, Sequence):
        data = data[k]
      elif not isinstance(k, Index) and isinstance(data, Mapping):
        data = data[k]
      else:
        raise ValueError(
            f'Cannot use {type(k)} ({k}) as a key for {type(data)} data.'
        )
    return self._maybe_map(data)

  def __getitem__(
      self, keys: TreeMapKey | TreeMapKeys
  ) -> MapLikeTree[LeafValueT] | None:
    """Returns value for a single key, returns a tuple for multi-key."""
    match keys:
      # Check key Path first to distinguish it from multi-key case below.
      case Key():
        return self.__get(keys)
      case ():
        return ()
      # Always treats the first tuple dimension as multi-key.
      case (_, *_):
        return tuple(self.__get(key) for key in keys)
      case _:
        return self.__get(keys)

  def _tree_iter(
      self,
      data: MapLikeTree[LeafValueT],
      parent_key_path: tuple[TreeMapKey, ...],
  ) -> Iterator[tuple[TreeMapKey, ...]]:
    """Iterates through the tree using DFS and yield all key Paths."""
    match data:
      case (_, *_):
        for i, v in enumerate(data):
          yield from self._tree_iter(v, parent_key_path + (Index(i),))
      case Mapping():
        for k, v in data.items():
          yield from self._tree_iter(v, parent_key_path + (k,))
        if not data:
          yield parent_key_path
      case _:
        # This captures empty list and non-list, non-dict pattern.
        yield parent_key_path

  def __iter__(self) -> Iterator[TreeMapKey]:
    keys = set()
    for key in self._tree_iter(self.data, ()):
      if key not in keys:
        keys.add(key)
        yield Key(key)

  def __len__(self) -> int:
    return len(list(iter(self)))

  def __str__(self):
    """Directly stringifies as the tree, no metadata is needed."""
    return str(self.data)

  def to_dict(self) -> dict[TreeMapKey, Any]:
    """Converts the tree to a flat dict with Path as the keys."""
    return {path: v for path, v in self.items()}

  def _set_by_path(
      self,
      tree: MapLikeTree | None,
      key_path: Key,
      value: Any,
  ) -> MapLikeTree[LeafValueT]:
    """Shallow copies the root and set tree[k] as "value" when applicable."""
    if not isinstance(key_path, Key):
      key_path = Key.new(key_path)
    # Empty Path means replacing the root directly with value.
    if key_path == Key() or _is_key(key_path[0], _SELF):
      return value
    if not tree:
      # The parent passes down None when the key was not found. Constructs the
      # tree according to the key_path when not strict.
      if self.strict:
        raise ValueError('Input tree cannot be empty when "strict" is True.')
      return _default_tree(key_path, value)

    # Not a mutable container, constructs the mutable counterpart first.
    container_type = None
    if not hasattr(tree, '__setitem__'):
      # Record its original type for later recovery of the same type.
      container_type = type(tree)
      match tree:
        case Sequence():
          result = list(tree)
        case Mapping():
          result = dict(tree)
        case _:
          # Remaining key cannot be resolved, treats this as not-found error.
          raise KeyError(f'{key_path} in {tree}')
    else:
      result = copy.copy(tree)

    # Setting the item.
    try:
      match key_path:
        case (Reserved() as reserved, *_):
          if _is_key(reserved, _SKIP):
            pass
        case (Index(key), *rest_keys) if isinstance(result, Sequence):
          if key == len(result):
            result.append(None)
          result[key] = self._set_by_path(result[key], Key(rest_keys), value)
        case (key, *rest_keys) if isinstance(result, Mapping):
          result[key] = self._set_by_path(
              result.get(key, None), Key(rest_keys), value
          )
        case _:
          raise ValueError(f'Unsupported key "{key_path}" for input of {tree}.')
    except (ValueError, KeyError, IndexError) as e:
      raise ValueError(
          f'Failed to insert {key_path}:{value} to \n{tree_shape(result)}'
      ) from e

    # Recovers the container type when applicable.
    if container_type is not None and self.consistent_type:
      try:
        result = container_type(result)  # pylint: disable=too-many-function-args
      except TypeError as e:
        raise ValueError(f'Cannot enforce {container_type} on {result}') from e
    return result

  # TODO: b/318463291 - adds in-place option for efficiency.
  def copy_and_set(
      self, keys: TreeMapKey | TreeMapKeys, values: Any
  ) -> TreeMapView[LeafValueT]:
    """Shallow copies the nodes along the path and set the leaf as the value."""
    # Normalizes the key to Path() and routes the correct way to call single
    # Path _copy_and_set.
    match keys:
      case Key():
        data = self._set_by_path(self.data, keys, values)
      case ():
        if values:
          raise ValueError(f'Keys cannot be empty: {keys=}')
        data = self.data
      # Multiple key cases (even single key within a tuple).
      case (_, *_):
        data = self.data
        values = values if isinstance(values, tuple) else (values,)
        for key, value in zip(keys, values, strict=True):
          data = self._set_by_path(data, key, value)
      case _:
        data = self._set_by_path(self.data, Key.new(keys), values)
    return dataclasses.replace(self, data=data)  # pylint: disable=undefined-variable

  def copy_and_update(self, other: Mapping[TreeMapKey, Any]) -> TreeMapView:
    """Copy and update the original tree given a map."""
    if other:
      return self.copy_and_set(*zip(*other.items(), strict=True))
    else:
      return self

  def copy_and_map(self, map_fn: Callable[..., Any]) -> MapLikeTree:
    """Copy and apply a map function to the tree."""
    view = TreeMapView.as_view(self, map_fn=map_fn)
    return TreeMapView().copy_and_update(view.to_dict())

  def __or__(self, other: Mapping[TreeMapKey, Any]) -> TreeMapView:
    """Alias for `copy_and_update`."""
    return self.copy_and_update(other)
