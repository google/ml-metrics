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
"""Base types used throughout the library."""

import abc
from typing import Any, Protocol, TypeGuard, TypeVar, runtime_checkable
from numpy import typing as npt

_T = TypeVar('_T')


@runtime_checkable
class Makeable(Protocol[_T]):
  """A config class that can make a Metric class."""

  @abc.abstractmethod
  def make(self) -> _T:
    """Makes a new Metric."""


@runtime_checkable
class Resolvable(Protocol[_T]):
  """All Resolvlables implements a `result_` to resolve the underlying value."""

  @abc.abstractmethod
  def result_(self) -> _T:
    """Interface to get the result of the underlying value."""


class Shardable(Protocol):
  """A sharded data source for chainables."""

  @abc.abstractmethod
  def shard(self, *args, **kwargs):
    """Iterates the data source given a shard index and number of shards."""


class Serializable(Protocol):
  """An object that can be both serialized and deserialized."""

  @abc.abstractmethod
  def get_config(self):
    """Gets the state of the object that can be used to recover the object."""

  @abc.abstractmethod
  def from_config(self, *args, **kwargs):
    """Iterates the data source given a shard index and number of shards."""


class Recoverable(Protocol):
  """An object that can be both serialized and deserialized."""

  state: Any

  @abc.abstractmethod
  def from_state(self, *args, **kwargs):
    """Recover from the state."""


MaybeResolvable = Resolvable[_T] | _T


class RandomAccessible(Protocol[_T]):

  def __getitem__(self, idx: int | slice) -> _T:
    """Same as Sequence.__getitem__."""

  def __len__(self) -> int:
    """Same as Sequence.__len__."""


def obj_has_method(obj: Any, method_name: str) -> bool:
  """Checks if the object has a method."""
  method = getattr(obj, method_name, False)
  return method and getattr(method, '__self__', None) is obj


def is_resolvable(obj: Resolvable[_T] | Any) -> TypeGuard[Resolvable[_T]]:
  """Checks if the object is a Resolvable."""
  return obj_has_method(obj, 'result_')


def is_makeable(obj: Makeable[_T] | Any) -> TypeGuard[Makeable[_T]]:
  """Checks if the object is a Makeable."""
  return obj_has_method(obj, 'make')


def is_shardable(obj: Shardable | Any) -> TypeGuard[Shardable]:
  """Checks if the object is a Shardable."""
  return obj_has_method(obj, 'shard')


def is_serializable(obj: Serializable | Any) -> TypeGuard[Serializable]:
  """Checks if the object is a Shardable."""
  return obj_has_method(obj, 'get_config') and obj_has_method(
      obj, 'from_config'
  )


def is_recoverable(obj: Recoverable | Any) -> TypeGuard[Recoverable]:
  """Checks if the object is a Shardable."""
  return obj_has_method(obj, 'from_state') and hasattr(obj, 'state')


def is_array_like(obj: list[Any] | tuple[Any, ...] | npt.ArrayLike) -> bool:
  """Checks if the object is an array-like object."""
  return isinstance(obj, (list, tuple)) or (
      hasattr(obj, '__array__') and obj.ndim > 0
  )
