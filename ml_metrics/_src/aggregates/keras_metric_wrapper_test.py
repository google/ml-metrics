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
"""Wrapper for Keras metrics."""

import importlib

from ml_metrics._src.aggregates import keras_metric_wrapper

from absl.testing import absltest


class MockKerasMetric:

  def __init__(self):
    self.reset_state()

  def reset_state(self):
    self._state = 0

  def update_state(self, inputs):
    self._state += sum(inputs)

  def merge_state(self, states):
    for state in states:
      self._state += state

  def result(self):
    return self._state


class KerasTest(absltest.TestCase):

  def test_mock_keras_metric(self):
    metric = keras_metric_wrapper.KerasAggregateFn(MockKerasMetric())
    self.assertEqual(6, metric([1, 2, 3]))

  def test_keras_metric_wrapper_merge(self):
    try:
      tf = importlib.import_module("tensorflow")
    except ImportError:
      # Ignores the import error if tensorflow is not installed.
      return
    metric1 = keras_metric_wrapper.KerasAggregateFn(
        tf.keras.metrics.Mean(name="mean")
    )
    metric2 = keras_metric_wrapper.KerasAggregateFn(
        tf.keras.metrics.Mean(name="mean")
    )
    state1, state2 = metric1.create_state(), metric2.create_state()
    merged_state = metric1.merge_states([
        metric1.update_state(state1, [1, 2, 3]),
        metric2.update_state(state2, [4, 5, 6]),
    ])
    self.assertEqual(3.5, metric1.get_result(merged_state).numpy())

  def test_keras_metric_wrapper(self):
    try:
      tf = importlib.import_module("tensorflow")
    except ImportError:
      # Ignores the import error if tensorflow is not installed.
      return
    metric = keras_metric_wrapper.KerasAggregateFn(
        tf.keras.metrics.Mean(name="mean")
    )
    self.assertEqual(2, metric([1, 2, 3]))


if __name__ == "__main__":
  absltest.main()
