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
"""Base types used throughout the library."""

import abc
from typing import Any, Protocol, TypeVar, runtime_checkable
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


MaybeResolvable = Resolvable[_T] | _T


def is_array_like(obj: list[Any] | tuple[Any, ...] | npt.ArrayLike) -> bool:
  """Checks if the object is an array-like object."""
  return isinstance(obj, (list, tuple)) or (
      hasattr(obj, '__array__') and obj.ndim > 0
  )
