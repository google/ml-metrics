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
          reference=['abc def', 'no match'],
          matcher=text.exact_match,
          expected_result=True,
      ),
      dict(
          testcase_name='exact_match_single_string',
          sample='abc def',
          reference='abc def',
          matcher=text.exact_match,
          expected_result=True,
      ),
      dict(
          testcase_name='exact_match_false',
          sample='abc def',
          reference=['no match'],
          matcher=text.exact_match,
          expected_result=False,
      ),
      dict(
          testcase_name='sample_startswith_reference_match',
          sample='abc',
          reference=['ab', 'no match'],
          matcher=text.sample_startswith_reference_match,
          expected_result=True,
      ),
      dict(
          testcase_name='sample_startswith_reference_match_single_string',
          sample='abc',
          reference='ab',
          matcher=text.sample_startswith_reference_match,
          expected_result=True,
      ),
      dict(
          testcase_name='sample_startswith_reference_match_false',
          sample='abc',
          reference=['no match'],
          matcher=text.sample_startswith_reference_match,
          expected_result=False,
      ),
      dict(
          testcase_name='reference_startswith_sample_match',
          sample='abc',
          reference=['abcd', 'no match'],
          matcher=text.reference_startswith_sample_match,
          expected_result=True,
      ),
      dict(
          testcase_name='reference_startswith_sample_match_single_string',
          sample='abc',
          reference='abcd',
          matcher=text.reference_startswith_sample_match,
          expected_result=True,
      ),
      dict(
          testcase_name='reference_startswith_sample_match_false',
          sample='abc',
          reference=['no match'],
          matcher=text.reference_startswith_sample_match,
          expected_result=False,
      ),
      dict(
          testcase_name='sample_in_reference_match',
          sample='bc',
          reference=['abcd', 'no match'],
          matcher=text.sample_in_reference_match,
          expected_result=True,
      ),
      dict(
          testcase_name='sample_in_reference_match_single_string',
          sample='bc',
          reference='abcd',
          matcher=text.sample_in_reference_match,
          expected_result=True,
      ),
      dict(
          testcase_name='sample_in_reference_match_false',
          sample='bc',
          reference=['no match'],
          matcher=text.sample_in_reference_match,
          expected_result=False,
      ),
      dict(
          testcase_name='reference_in_sample_match',
          sample='abc',
          reference=['bc', 'no match'],
          matcher=text.reference_in_sample_match,
          expected_result=True,
      ),
      dict(
          testcase_name='reference_in_sample_match_single_string',
          sample='abc',
          reference='bc',
          matcher=text.reference_in_sample_match,
          expected_result=True,
      ),
      dict(
          testcase_name='reference_in_sample_match_false',
          sample='abc',
          reference=['no match'],
          matcher=text.reference_in_sample_match,
          expected_result=False,
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

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_text',
          text_input='Hello. How are you?',
          expected_count=3.5,
      ),
      dict(
          testcase_name='empty_text',
          text_input='',
          expected_count=0.0,
      ),
      dict(
          testcase_name='no_word',
          text_input='., ? !',
          expected_count=0.0,
      ),
  )
  def test_average_word_length(self, text_input, expected_count):
    self.assertAlmostEqual(expected_count, text.average_word_length(text_input))

  @parameterized.named_parameters(
      dict(
          testcase_name='token_match_perfect',
          sample='hello world',
          reference='hello world',
          expected_result=1.0,
      ),
      dict(
          testcase_name='token_match_partial_with_longer_sample',
          sample='hello a world',
          reference='hello world',
          expected_result=1.0 / 3.0,
      ),
      dict(
          testcase_name='token_match_partial_with_longer_reference',
          sample='hello world',
          reference='hello a world',
          expected_result=1.0 / 3.0,
      ),
  )
  def test_token_match_rate(self, sample, reference, expected_result):
    def tokenizer(x):
      return x.split(' ')

    self.assertAlmostEqual(
        expected_result,
        text.token_match_rate(sample, reference, tokenizer),
    )


if __name__ == '__main__':
  absltest.main()
