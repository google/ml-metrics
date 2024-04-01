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
from ml_metrics._src.scores import text
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


if __name__ == '__main__':
  absltest.main()
