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
from ml_metrics._src.metrics import text

from absl.testing import absltest


class TextTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count',
          metric=text.avg_char_count,
      ),
      dict(
          testcase_name='avg_word_count',
          metric=text.avg_word_count,
      ),
      dict(
          testcase_name='avg_token_count',
          metric=text.avg_token_count,
          tokenizer=lambda x: x.split(),  # Dummy tokenizer
      ),
  ])
  def test_metrics_empty(self, *, metric, **kwargs):
    print('kwargs:', kwargs)
    result = metric([], **kwargs)
    self.assertAlmostEqual(0, result)

  @parameterized.named_parameters([
      dict(
          testcase_name='avg_char_count',
          metric=text.avg_char_count,
          exptected_result=5/2,
      ),
      dict(
          testcase_name='avg_word_count',
          metric=text.avg_word_count,
          exptected_result=3/2,
      ),
  ])
  def test_simple_metrics(self, metric, exptected_result):
    texts = ['abc', 'd e!', None]
    result = metric(texts)
    self.assertAlmostEqual(exptected_result, result)

  def test_avg_token_count(self):
    texts = ['aba xaba', 'ababa', None]
    # A tokenizer designed to identify and return the positions of the pattern
    # 'aba' within a text string.
    tokenizer = lambda x: list(re.findall(r'(?=(aba))', x))
    result = text.avg_token_count(texts, tokenizer=tokenizer)
    expected_result = 4/2
    self.assertAlmostEqual(expected_result, result)


if __name__ == '__main__':
  absltest.main()
