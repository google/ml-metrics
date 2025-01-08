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
from typing import Any, Iterable, Protocol, TypeGuard, TypeVar, runtime_checkable
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


class Shardable(Protocol[_T]):
  """A sharded data source for chainables."""

  @abc.abstractmethod
  def iter_shard(self, shard_index: int = 0) -> Iterable[_T]:
    """Iterates the data source given a shard index and number of shards."""


MaybeResolvable = Resolvable[_T] | _T


def is_resolvable(obj: Resolvable[_T] | Any) -> TypeGuard[Resolvable[_T]]:
  """Checks if the object is a Resolvable."""
  result_ = getattr(obj, 'result_', None)
  # Also distinguish between classmethod or instance method.
  return result_ and getattr(result_, '__self__', None) is obj


def is_makeable(obj: Makeable[_T] | Any) -> TypeGuard[Makeable[_T]]:
  """Checks if the object is a Makeable."""
  make = getattr(obj, 'make', None)
  # Also distinguish between classmethod or instance method.
  return make and getattr(make, '__self__', None) is obj


def is_shardable(obj: Shardable[_T] | Any) -> TypeGuard[Shardable[_T]]:
  """Checks if the object is a Shardable."""
  shardable = getattr(obj, 'shard', None)
  return shardable and getattr(shardable, '__self__', None) is obj


def is_array_like(obj: list[Any] | tuple[Any, ...] | npt.ArrayLike) -> bool:
  """Checks if the object is an array-like object."""
  return isinstance(obj, (list, tuple)) or (
      hasattr(obj, '__array__') and obj.ndim > 0
  )
