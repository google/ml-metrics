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
"""Aggregation interfaces."""

# pylint: disable=g-importing-member
# pylint: disable=unused-import
from ml_metrics._src.aggregates.base import Aggregatable
from ml_metrics._src.aggregates.base import as_agg_fn
from ml_metrics._src.aggregates.base import CallableMetric
from ml_metrics._src.aggregates.base import Metric
from ml_metrics._src.aggregates.keras_metric_wrapper import is_keras_metric
from ml_metrics._src.aggregates.keras_metric_wrapper import KerasAggregateFn
from ml_metrics._src.aggregates.types import AverageType
from ml_metrics._src.aggregates.types import InputType
