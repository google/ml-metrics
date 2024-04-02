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
"""Example Count Metric."""

from collections.abc import Iterable
from typing import Any

from ml_metrics._src.aggregates import example_count as _example_count


_METRIC_PYDOC_POSTFIX = """

  Args:
    examples: The examples.
    example_weights: (Optional) the example weights.

  Returns:
    The weighted example count.
"""


def example_count(
    examples: Iterable[Any],
    example_weights: Iterable[float] | None = None,
) -> float:
  """Compute F1 Score classification metric."""
  return _example_count.ExampleCountAggFn()(
      examples=examples, example_weights=example_weights
  )


example_count.__doc__ += _METRIC_PYDOC_POSTFIX
