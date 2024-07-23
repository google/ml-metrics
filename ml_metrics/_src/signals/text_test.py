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

from absl.testing import parameterized
from ml_metrics._src.signals import text
from absl.testing import absltest


class TextTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='simple_text',
          input_text='aBc',
          expected_counts=3,
      ),
      dict(
          testcase_name='empty_text',
          input_text='',
          expected_counts=0,
      ),
      dict(
          testcase_name='mixed_char_text',
          input_text='a b!@ 14cD.',
          expected_counts=4,
      ),
  ])
  def test_alphabetical_char_count(self, input_text, expected_counts):
    count = text.alphabetical_char_count(input_text)
    self.assertEqual(expected_counts, count)

  @parameterized.named_parameters([
      dict(
          testcase_name='simple_text',
          input_text='aBc def',
          expected_counts=2,
      ),
      dict(
          testcase_name='empty_text',
          input_text='',
          expected_counts=0,
      ),
      dict(
          testcase_name='mixed_symbol_text',
          input_text='1. I\'m doing well. How are you?',
          expected_counts=6,
      ),
  ])
  def test_word_count(self, input_text, expected_counts):
    count = text.word_count(input_text)
    self.assertEqual(expected_counts, count)

  def test_token_count(self):
    def tokenizer(x):
      return [x[:i+1] for i in range(len(x))]

    count = text.token_count(text='abcd', tokenizer=tokenizer)
    self.assertEqual(4, count)

  @parameterized.named_parameters([
      dict(
          testcase_name='exact_match',
          sample='abc def',
          reference='abc def',
          matcher=text.exact_match,
          expected_result=True,
      ),
      dict(
          testcase_name='sample_startswith_reference_match',
          sample='abc',
          reference='ab',
          matcher=text.sample_startswith_reference_match,
          expected_result=True,
      ),
      dict(
          testcase_name='reference_startswith_sample_match',
          sample='abc',
          reference='abcd',
          matcher=text.reference_startswith_sample_match,
          expected_result=True,
      ),
      dict(
          testcase_name='sample_in_reference_match',
          sample='bc',
          reference='abcd',
          matcher=text.sample_in_reference_match,
          expected_result=True,
      ),
      dict(
          testcase_name='reference_in_sample_match',
          sample='abc',
          reference='bc',
          matcher=text.reference_in_sample_match,
          expected_result=True,
      ),
  ])
  def test_matchers(self, sample, reference, matcher, expected_result):
    self.assertEqual(expected_result, matcher(sample, reference))

  def test_non_ascii_char_count(self):
    count = text.non_ascii_char_count(text='Hällö, wörld!')
    # Non-ascii characters are "Hll, wrld!"
    self.assertEqual(10, count)

  def test_is_all_whitespace(self):
    self.assertTrue(text.is_all_whitespace(text=''))
    self.assertTrue(text.is_all_whitespace(text=' \n\t'))
    self.assertFalse(text.is_all_whitespace(text='abc'))


if __name__ == '__main__':
  absltest.main()
