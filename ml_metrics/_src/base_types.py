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
import enum
from typing import Protocol, TypeVar, runtime_checkable

MakeT = TypeVar('MakeT')


@runtime_checkable
class Makeable(Protocol[MakeT]):
  """A config class that can make a Metric class."""

  @abc.abstractmethod
  def make(self) -> MakeT:
    """Makes a new Metric."""


# TODO: b/312290886 - move this to Python StrEnum when moved to Python 3.11.
class StrEnum(str, enum.Enum):
  """Enum where members also must be strings."""

  __str__ = str.__str__

  __repr__ = str.__repr__

  __format__ = str.__format__

  __iter__ = enum.Enum.__iter__
