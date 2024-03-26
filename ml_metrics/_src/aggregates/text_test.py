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
"""Tests for text."""

import re

from absl.testing import parameterized
from ml_metrics._src.aggregates import text
from ml_metrics._src.aggregates import utils

from absl.testing import absltest


class SimpleMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count_metric',
          metric=text.AvgCharCount,
          item_count=5,
          existing_text_count=2,
      ),
      dict(
          testcase_name='avg_word_count_metric',
          metric=text.AvgWordCount,
          item_count=3,
          existing_text_count=2,
      ),
  ])
  def test_compute_metric(
      self, metric, item_count, existing_text_count
  ):
    batch = ['abc', 'd e!']
    avg_item_metric = metric()
    batch_result = avg_item_metric.add(batch)

    self.assertAlmostEqual(
        batch_result, item_count / existing_text_count
    )
    expected_state = utils.MeanState(item_count, existing_text_count)
    self.assertEqual(avg_item_metric.state, expected_state)
    self.assertAlmostEqual(
        avg_item_metric.result(), item_count / existing_text_count
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count_metric',
          metric=text.AvgCharCount,
          batch_result=2,
          item_count=7,
          existing_text_count=3,
      ),
      dict(
          testcase_name='avg_word_count_metric',
          metric=text.AvgWordCount,
          batch_result=1,
          item_count=4,
          existing_text_count=3,
      ),
  ])
  def test_avg_char_count_metric_add(
      self, metric, batch_result, item_count, existing_text_count
  ):
    avg_item_metric = metric()

    batch_0 = ['abc', 'd e!']
    avg_item_metric.add(batch_0)

    batch_1 = ['fi']
    batch_1_result = avg_item_metric.add(batch_1)
    self.assertAlmostEqual(batch_1_result, batch_result)

    expected_updated_state = utils.MeanState(item_count, existing_text_count)
    self.assertEqual(avg_item_metric.state, expected_updated_state)
    self.assertAlmostEqual(
        avg_item_metric.result(), item_count / existing_text_count
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count_metric',
          metric=text.AvgCharCount,
          item_count=7,
          existing_text_count=3,
      ),
      dict(
          testcase_name='avg_word_count_metric',
          metric=text.AvgWordCount,
          item_count=3,
          existing_text_count=3,
      ),
  ])
  def test_avg_char_count_metric_merge(
      self, metric, item_count, existing_text_count
  ):
    batch_0 = ['abc', 'de']
    avg_item_metric_0 = metric()
    avg_item_metric_0.add(batch_0)

    batch_1 = ['fi']
    avg_item_metric_1 = metric()
    avg_item_metric_1.add(batch_1)

    avg_item_metric_0.merge(avg_item_metric_1)

    expected_state = utils.MeanState(item_count, existing_text_count)
    self.assertEqual(avg_item_metric_0.state, expected_state)
    self.assertAlmostEqual(
        avg_item_metric_0.result(), item_count / existing_text_count
    )


class TokenizeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A tokenizer designed to identify and return the positions of the pattern
    # 'aba' within a text string.
    self._tokenizer = lambda x: list(re.findall(r'(?=(aba))', x))

  def test_compute_tokenize_metric(self):
    batch = ['aba xaba', 'ababa', None]
    tokenize_metric = text.Tokenize(tokenizer=self._tokenizer)
    batch_result = tokenize_metric.add(batch)

    self.assertAlmostEqual(batch_result, 4/2)
    expected_state = utils.MeanState(4, 2)
    self.assertEqual(tokenize_metric.state, expected_state)
    self.assertAlmostEqual(tokenize_metric.result(), 4/2)

  def test_tokenize_metric_add(self):
    tokenize_metric = text.Tokenize(tokenizer=self._tokenizer)

    batch_0 = ['aba xaba', 'ababa']
    tokenize_metric.add(batch_0)

    batch_1 = ['xaba']
    batch_result = tokenize_metric.add(batch_1)
    self.assertAlmostEqual(batch_result, 1)

    expected_updated_state = utils.MeanState(5, 3)
    self.assertEqual(tokenize_metric.state, expected_updated_state)
    self.assertAlmostEqual(tokenize_metric.result(), 5/3)

  def test_tokenize_metric_merge(self):
    batch_0 = ['aba xaba', 'ababa']
    tokenize_metric_0 = text.Tokenize(tokenizer=self._tokenizer)
    tokenize_metric_0.add(batch_0)

    batch_1 = ['xaba']
    tokenize_metric_1 = text.Tokenize(tokenizer=self._tokenizer)
    tokenize_metric_1.add(batch_1)

    tokenize_metric_0.merge(tokenize_metric_1)

    expected_state = utils.MeanState(5, 3)
    self.assertEqual(tokenize_metric_0.state, expected_state)
    self.assertAlmostEqual(tokenize_metric_0.result(), 5/3)

if __name__ == '__main__':
  absltest.main()
