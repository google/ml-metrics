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
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
import copy
import dataclasses
import functools
from typing import Any, Protocol, Self, TypeVar, Union, overload

from ml_metrics._src import base_types
import numpy as np


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


class Literal:
  """Convenient class to identify literal values."""

  def __init__(self, value: Any):
    self._value = value

  @property
  def value(self):
    return self._value

  def __repr__(self):
    return f'Literal({repr(self._value)})'


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


class NullMap(MapLike):
  """An empty MapLike to be used as a placeholder."""

  def __getitem__(self, key: Any):
    return


# A MapLike tree is a nested MapLike structure. E.g., Sequence and Mapping
# DataFrame, Numpy array are all MapLike.
# Uses Union here since forward reference does not work with |.
MapLikeTree = MapLike[BaseKey, Union['MapLikeTree', LeafValueT]]
MapLikeTreeCallable = Callable[[MapLikeTree | None], MapLikeTree | None]


DEFAULT_FILTER = 'DEFAULT_FILTER'


def apply_mask(
    items: MapLikeTree[Any],
    *,
    masks: MapLikeTree[bool],
    replace_false_with: Any = DEFAULT_FILTER,
):
  """Applies masks to inputs.

  The masks and the inputs have to have the tree structure and shape, i.e.,
  there has to be a boolean per tree of elements or element in the inputs.

  Args:
    items: A tree of inputs.
    masks: A tree of masks.
    replace_false_with: The value to replace False with when mask_behavior is
      "replace". If not set, throw away the elements that are False in the
      masks. If set, replace the elements that are False in the masks with the
      value of replace_false_with.

  Returns:
    A tree of inputs with the masks applied.
  """
  # Test comparison to literal boolean True, note, this includs 1.
  if not base_types.is_array_like(masks) and masks == True:  # pylint: disable=singleton-comparison
    return items
  # array like masks and items, this includes list, tuple, np array.
  elif base_types.is_array_like(items) and base_types.is_array_like(masks):
    if hasattr(masks, '__array__') and getattr(masks, 'dtype') == bool:
      if replace_false_with != DEFAULT_FILTER:
        return np.where(masks, items, replace_false_with)
      else:
        return np.asarray(items)[masks]
    result = []
    for elem, mask in zip(items, masks, strict=True):
      if mask == True:  # pylint: disable=singleton-comparison
        result.append(elem)
      elif mask == False:  # pylint: disable=singleton-comparison
        if replace_false_with != DEFAULT_FILTER:
          result.append(replace_false_with)
      else:
        result.append(
            apply_mask(
                elem,
                masks=mask,
                replace_false_with=replace_false_with,
            )
        )
    if isinstance(items, tuple):
      result = tuple(result)
    # For data that is array-like, will be converted into a np array.
    elif hasattr(items, '__array__'):
      if hasattr(result, 'dtype'):
        result = np.asarray(result, dtype=hasattr(items, 'dtype'))
      else:
        result = np.asarray(result)
  elif isinstance(masks, dict) and isinstance(items, dict):
    result = {}
    for key, mask in masks.items():
      value = items.get(key)
      if mask == True:  # pylint: disable=singleton-comparison
        result[key] = value
      elif mask == False:  # pylint: disable=singleton-comparison
        if replace_false_with != DEFAULT_FILTER:
          result[key] = replace_false_with
      else:
        result[key] = apply_mask(
            value,
            masks=mask,
            replace_false_with=replace_false_with,
        )
  # When masks is not a dict, try to apply the mask to the leaf elements of the
  # dict. Note that TreeMapView will only apply the mask lazily when a leaf
  # element is accessed.
  elif base_types.is_array_like(masks) and isinstance(items, dict):
    return TreeMapView.as_view(
        items,
        map_fn=functools.partial(
            apply_mask,
            masks=masks,
            replace_false_with=replace_false_with,
        ),
    )
  else:
    raise ValueError(
        'Masks and inputs have to be of types (dict, dict), (array, array), '
        f'(array, dict), or (True, Any) got: {type(masks)=}, {type(items)=}'
    )
  return result


def _is_key(key, other_key):
  return isinstance(key, type(other_key)) and key == other_key


class Key(tuple[BaseKey, ...]):
  """Convenient subtype of a tuple for a key path.

  This is immutable, and can only be called by either `new('a', 'b')`
  method or inputs a tuple: Path(('a', 'b')). For convenience, user can also
  attach individual keys by directly referencing the string name. E.g.,
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
  def Literal(cls, value: Any):  # pylint: disable=invalid-name
    return Literal(value)

  @classmethod
  def new(cls, *args: tuple[BaseKey, ...]) -> Self:
    return cls(tuple(args))

  def __getattr__(self, name: str):
    return Key(self + (name,))

  def at(self, key: BaseKey):
    return Key(self + (key,))

  def __repr__(self):
    return f'Path{super().__repr__()}'


TreeMapKey = BaseKey | Key | Reserved | Literal
# For keys, if there is tuple, the first dimension is always keys dimension.
TreeMapKeys = tuple[TreeMapKey, ...] | Mapping[str, TreeMapKey]


def normalize_keys(keys: TreeMapKey | TreeMapKeys) -> TreeMapKeys:
  """Normalizes the keys into a tuple of keys."""
  if isinstance(keys, Key) or not isinstance(keys, (tuple, list)):
    # Note: this has to be before tuple since Key is a subclass of tuple.
    return (keys,)
  elif isinstance(keys, (list, tuple)):
    return tuple(keys)
  else:
    raise TypeError(f'Unsupported keys {keys}')


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
    key_paths: A sequence of key paths to be used when iterating the tree.
    map_fn: A function that will be applied to the leaf value when accessing the
      tree.
    strict: If True, non-existing key will cause a KeyError.
  """

  data: MapLikeTree = dataclasses.field(default_factory=NullMap)
  key_paths: tuple[TreeMapKey, ...] | None = None
  map_fn: Callable[..., Any] | None = dataclasses.field(
      kw_only=True,
      default=None,
  )
  strict: bool = dataclasses.field(kw_only=True, default=False)

  @classmethod
  def as_view(
      cls,
      tree_or_view: MapLikeTree[LeafValueT] | TreeMapView[LeafValueT],
      key_paths: tuple[TreeMapKey, ...] | None = None,
      map_fn: Callable[..., Any] | None = None,
  ) -> TreeMapView:
    """Util to use a MapLikeTree as a Map."""
    if not isinstance(tree_or_view, TreeMapView):
      tree_or_view = TreeMapView(tree_or_view)
    if map_fn is not None:
      tree_or_view = dataclasses.replace(
          tree_or_view,
          map_fn=map_fn,
      )
    if key_paths is not None:
      tree_or_view = dataclasses.replace(
          tree_or_view,
          key_paths=key_paths,
      )
    return tree_or_view

  def _maybe_map(self, data):
    return self.map_fn(data) if self.map_fn else data

  def __get(self, key: TreeMapKey) -> LeafValueT:
    """Gets the value from a single Key or Path."""
    key = key if isinstance(key, Key) else Key((key,))
    data = self.data
    for k in key:
      if _is_key(k, Key.SELF):
        return self._maybe_map(data)
      if isinstance(k, Literal):
        return k.value
      if isinstance(k, Index) and base_types.is_array_like(data):
        data = data[k]
      elif not isinstance(k, Index) and isinstance(data, Mapping):
        data = data[k]
      else:
        raise ValueError(
            f'Cannot use {type(k)} ({k}) as a key for {type(data)} data.'
        )
    return self._maybe_map(data)

  @overload
  def __getitem__(self, keys: TreeMapKey) -> LeafValueT:
    """Returns value for a single key, returns a tuple for multi-key."""
    ...

  @overload
  def __getitem__(self, keys: tuple[TreeMapKey, ...]) -> tuple[LeafValueT, ...]:
    ...

  def __getitem__(self, keys):
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
    # Uses user specified key_paths if available. Otherwise, DFS and yield all
    # key paths.
    if self.key_paths is not None:
      yield from self.key_paths
    else:
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

  def _set_by_path(
      self,
      tree: MapLikeTree,
      key_path: Key,
      value: LeafValueT,
      in_place: bool = False,
  ) -> MapLikeTree[LeafValueT]:
    """Shallow copies the root and set tree[k] as "value" when applicable."""
    if not isinstance(key_path, Key):
      key_path = Key.new(key_path)
    # Empty Path means replacing the root directly with value at the callsite.
    if key_path == Key() or _is_key(key_path[0], _SELF):
      return value

    # Not a mutable container, constructs the mutable counterpart first.
    container_maker = None
    if isinstance(tree, NullMap):
      if self.strict:
        raise ValueError('Input tree cannot be empty when "strict" is True.')
      return _default_tree(key_path, value)
    elif not hasattr(tree, '__setitem__'):
      # Returns a copy of a tuple from internals.
      if not in_place and isinstance(tree, tuple):
        container_maker = tuple
        result = list(tree)
      else:
        raise TypeError(f'Insert to immutable {type(tree)} with {key_path}.')
    else:
      result = tree if in_place else copy.copy(tree)

    # Setting the item.
    try:
      match key_path:
        case (Reserved() as reserved, *_):
          if _is_key(reserved, _SKIP):
            pass
        case (Index(key), *rest_keys) if base_types.is_array_like(result):
          if key == len(result):
            assert isinstance(result, list)
            result.append(NullMap())
          result[key] = self._set_by_path(
              result[key], Key(rest_keys), value, in_place
          )
        case (key, *rest_keys) if isinstance(result, Mapping):
          result[key] = self._set_by_path(
              result.get(key, NullMap()), Key(rest_keys), value, in_place
          )
        case _:
          raise ValueError(
              f'Unsupported key "{key_path}" for input of {repr(tree)}.'
          )
    except (ValueError, KeyError, IndexError) as e:
      raise ValueError(
          f'Failed to insert {key_path}:{value} to \n{tree_shape(result)}'
      ) from e

    # Recovers the container type when applicable.
    result = result if container_maker is None else container_maker(result)
    return result

  def set(
      self, keys: TreeMapKey | TreeMapKeys, values: Any, in_place: bool = True
  ) -> TreeMapView[LeafValueT]:
    """Shallow copies the nodes along the path and set the leaf as the value."""
    # Normalizes the key to Path() and routes the correct way to call single
    # Path _copy_and_set.
    match keys:
      case Key():
        data = self._set_by_path(self.data, keys, values, in_place)
      case ():
        if values:
          raise ValueError(f'Keys cannot be empty: {keys=}')
        data = self.data
      # Multiple key cases (even single key within a tuple).
      case (_, *_):
        data = self.data
        # Check exact type match because function multiple returns is a tuple.
        # e.g., NamedTuple won't be considered multiple outputs.
        values = values if type(values) is tuple else (values,)  # pylint: disable=unidiomatic-typecheck
        for key, value in zip(keys, values, strict=True):
          data = self._set_by_path(data, key, value, in_place)
      case _:
        data = self._set_by_path(self.data, Key.new(keys), values, in_place)
    return self if in_place else dataclasses.replace(self, data=data)

  def __setitem__(self, keys: TreeMapKey | TreeMapKeys, values: Any):
    self.set(keys, values)

  def keys(self):
    return tuple(k for k in self)

  def values(self):
    return tuple(self[k] for k in self)

  def copy_and_set(
      self,
      keys: TreeMapKey | TreeMapKeys,
      values: Any,
  ) -> TreeMapView[LeafValueT]:
    return self.set(keys, values, in_place=False)

  def copy_and_update(
      self,
      other: Mapping[TreeMapKey, Any] | Iterable[tuple[TreeMapKey, Any]],
  ) -> TreeMapView:
    """Copy and update the original tree given a map."""
    if not other:
      return self
    # tuple of key value pairs.
    if isinstance(other, Mapping):
      return self.copy_and_set(*zip(*other.items(), strict=True))
    else:
      return self.copy_and_set(*zip(*other))

  def __or__(self, other: Mapping[TreeMapKey, Any]) -> TreeMapView:
    """Alias for `copy_and_update`."""
    return self.copy_and_update(other)

  # TODO: b/311207032 - make the container type also consistent across the tree
  # with the original self.data.
  def apply(self) -> MapLikeTree[LeafValueT]:
    """Copy and apply a map function to the tree."""
    if self.map_fn is None and self.key_paths is None:
      return self.data
    initial_map = TreeMapView()
    if isinstance(self.data, tuple):
      initial_map = TreeMapView(())
    return initial_map.copy_and_update(self.items()).data
