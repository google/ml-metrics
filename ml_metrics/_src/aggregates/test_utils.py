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
"""test_utils."""
from ml_metrics._src.aggregates import base


class _SumMetric(base.CallableMetric):
  """Mock Metric for test."""

  def __init__(self, state=0):
    self._state = state

  def as_agg_fn(self):
    return base.as_agg_fn(self.__class__)

  @property
  def state(self):
    return self._state

  def new(self, x):
    return _SumMetric(state=sum(x))

  def merge(self, other):
    self._state += other.state

  def result(self):
    return self._state


class _SumAggFn:
  """Mock CombineFn for test."""

  def create_state(self):
    return 0

  def update_state(self, state, x):
    return state + sum(x)

  def merge_states(self, states):
    return sum(states)

  def get_result(self, state):
    return state
