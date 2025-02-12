# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for generating DataFrames."""

from __future__ import annotations

import collections
from typing import Any

from ml_metrics._src.chainables import transform
import pandas as pd


_METRIC_NAME = 'metric_name'
_SLICE = 'slice'
_VALUE = 'value'


def _first_or_tuple(x: tuple[Any, ...]) -> tuple[Any, ...] | Any:
  if isinstance(x, tuple) and len(x) == 1:
    return x[0]
  return x


_StrOrMetricKey = transform.MetricKey | str


def metrics_to_df(metrics: dict[_StrOrMetricKey, Any]) -> pd.DataFrame:
  """Converts the aggregation result to a DataFrame.

  This always converts the dict aggregation result to a DataFrame with
  the following columns:

    - metric_name: the name of the metric.
    - slice: the slice of the metric, if a slice is not specified, it will be
      'overall'.
    - value: the value of the metric.

  Args:
    metrics: the aggregation result.

  Returns:
    A DataFrame with the above columns.
  """
  sliced_results = collections.defaultdict(list)
  for k, v in metrics.items():
    if isinstance(k, str):
      sliced_results[_METRIC_NAME].append(k)
      sliced_results[_SLICE].append('overall')
      sliced_results[_VALUE].append(v)
    elif isinstance(k, transform.MetricKey):
      sliced_results[_METRIC_NAME].append(k.metrics)
      slice_name = _first_or_tuple(k.slice.features)
      slice_value = _first_or_tuple(k.slice.values)
      sliced_results[_SLICE].append(f'{slice_name}={slice_value}')
      sliced_results[_VALUE].append(v)
  return pd.DataFrame(sliced_results)
