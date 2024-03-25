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

from absl.testing import parameterized
from ml_metrics._src.aggregates import nlp
from ml_metrics._src.aggregates import utils

from absl.testing import absltest


class SimpleMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count_metric',
          maker=nlp.AvgCharCountMaker,
          item_count=5,
          existing_text_count=2,
      ),
      dict(
          testcase_name='avg_word_count_metric',
          maker=nlp.AvgWordCountMaker,
          item_count=3,
          existing_text_count=2,
      ),
  ])
  def test_compute_metric(
      self, maker, item_count, existing_text_count
  ):
    batch = ['abc', 'd e!']
    avg_item_metric = maker().make()
    batch_result = avg_item_metric.add(batch)

    self.assertAlmostEqual(
        batch_result, float(item_count / existing_text_count)
    )
    expected_state = utils.MeanState(item_count, existing_text_count)
    self.assertEqual(avg_item_metric.state, expected_state)
    self.assertAlmostEqual(
        avg_item_metric.result(), float(item_count / existing_text_count)
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count_metric',
          maker=nlp.AvgCharCountMaker,
          batch_result=float(2/1),
          item_count=7,
          existing_text_count=3,
      ),
      dict(
          testcase_name='avg_word_count_metric',
          maker=nlp.AvgWordCountMaker,
          batch_result=float(1/1),
          item_count=4,
          existing_text_count=3,
      ),
  ])
  def test_avg_char_count_metric_add(
      self, maker, batch_result, item_count, existing_text_count
  ):
    avg_item_metric = maker().make()

    batch_0 = ['abc', 'd e!']
    avg_item_metric.add(batch_0)

    batch_1 = ['fi']
    batch_1_result = avg_item_metric.add(batch_1)
    self.assertAlmostEqual(batch_1_result, batch_result)

    expected_updated_state = utils.MeanState(item_count, existing_text_count)
    self.assertEqual(avg_item_metric.state, expected_updated_state)
    self.assertAlmostEqual(
        avg_item_metric.result(), float(item_count / existing_text_count)
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count_metric',
          maker=nlp.AvgCharCountMaker,
          item_count=7,
          existing_text_count=3,
      ),
      dict(
          testcase_name='avg_word_count_metric',
          maker=nlp.AvgWordCountMaker,
          item_count=3,
          existing_text_count=3,
      ),
  ])
  def test_avg_char_count_metric_merge(
      self, maker, item_count, existing_text_count
  ):
    batch_0 = ['abc', 'de']
    avg_item_metric_0 = maker().make()
    avg_item_metric_0.add(batch_0)

    batch_1 = ['fi']
    avg_item_metric_1 = maker().make()
    avg_item_metric_1.add(batch_1)

    avg_item_metric_0.merge(avg_item_metric_1)

    expected_state = utils.MeanState(item_count, existing_text_count)
    self.assertEqual(avg_item_metric_0.state, expected_state)
    self.assertAlmostEqual(
        avg_item_metric_0.result(), float(item_count / existing_text_count)
    )


if __name__ == '__main__':
  absltest.main()
