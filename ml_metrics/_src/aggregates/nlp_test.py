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
"""Tests for nlp."""

from ml_metrics._src.aggregates import nlp
from ml_metrics._src.aggregates import utils

from absl.testing import absltest


class NlpTest(absltest.TestCase):

  def test_compute_avg_char_count_metric(self):
    batch = ['abc', 'd e!']
    avg_char_metric = nlp.AvgCharCountMaker().make()
    batch_result = avg_char_metric.add(batch)

    self.assertAlmostEqual(batch_result, float(5/2))
    expected_state = utils.MeanState(5, 2)
    self.assertEqual(avg_char_metric.state, expected_state)
    self.assertAlmostEqual(avg_char_metric.result(), float(5/2))

  def test_avg_char_count_metric_add(self):
    avg_char_metric = nlp.AvgCharCountMaker().make()

    batch_0 = ['abc', 'd e!']
    avg_char_metric.add(batch_0)

    batch_1 = ['fi']
    batch_result = avg_char_metric.add(batch_1)
    self.assertAlmostEqual(batch_result, float(2/1))

    expected_updated_state = utils.MeanState(7, 3)
    self.assertEqual(avg_char_metric.state, expected_updated_state)
    self.assertAlmostEqual(avg_char_metric.result(), float(7/3))

  def test_avg_char_count_metric_merge(self):
    batch_0 = ['abc', 'de']
    avg_char_metric_0 = nlp.AvgCharCountMaker().make()
    avg_char_metric_0.add(batch_0)

    batch_1 = ['fi']
    avg_char_metric_1 = nlp.AvgCharCountMaker().make()
    avg_char_metric_1.add(batch_1)

    avg_char_metric_0.merge(avg_char_metric_1)

    expected_state = utils.MeanState(7, 3)
    self.assertEqual(avg_char_metric_0.state, expected_state)
    self.assertAlmostEqual(avg_char_metric_0.result(), float(7/3))


if __name__ == '__main__':
  absltest.main()
