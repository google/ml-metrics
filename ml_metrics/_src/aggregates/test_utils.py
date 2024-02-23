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
