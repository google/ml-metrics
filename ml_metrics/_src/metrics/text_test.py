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
from ml_metrics._src.metrics import text
from absl.testing import absltest


class TextTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='topk_ngrams_first',
          metric=text.topk_ngrams_first,
          expected_result=[
              ('a b', 3/6),
              ('b c', 2/6),
          ],
      ),
      dict(
          testcase_name='topk_ngrams_all',
          metric=text.topk_ngrams_all,
          expected_result=[
              ('a b', 3/6),
              ('b c', 3/6),
          ],
      ),
  )
  def test_topk_ngrams(self, metric, expected_result):
    texts = [
        'a b c d',
        'a b',
        'a b',
        'b c',
        'b c d',
        'c d',
        None,
    ]
    result = metric(texts=texts, k=2, n=2)
    self.assertSequenceAlmostEqual(expected_result, result)

  @parameterized.named_parameters(
      dict(
          testcase_name='topk_ngrams_first',
          metric=text.topk_ngrams_first,
      ),
      dict(
          testcase_name='topk_ngrams_all',
          metric=text.topk_ngrams_all,
      ),
  )
  def test_topk_ngrams_empty(self, metric):
    result = metric(texts=[], k=2, n=2)
    self.assertSequenceAlmostEqual([], result)


if __name__ == '__main__':
  absltest.main()
