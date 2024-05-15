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


class TopKWordNgramsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='distinct_ngrams',
          k=2,
          n=2,
          use_first_ngram_only=False,
          count_duplicate=False,
          expected_result=[
              ('a a', 1 / 3),
              ('b b', 1 / 3),
              # ('c c', 1 / 3),
              # ('d a', 1 / 3),
          ],
      ),
      dict(
          testcase_name='count_duplicate',
          k=2,
          n=2,
          use_first_ngram_only=False,
          count_duplicate=True,
          expected_result=[
              ('b b', 2 / 3),
              ('a a', 1 / 3),
              # ('c c', 1 / 3),
              # ('d a', 1 / 3),
          ],
      ),
      dict(
          testcase_name='use_first_ngram_only',
          k=2,
          n=2,
          use_first_ngram_only=True,
          # count_duplicate will be ignored. See TopKWordNGrams description.
          count_duplicate=True,
          expected_result=[
              ('b b', 1 / 3),
              ('c c', 1 / 3),
              # ('d a', 1 / 3),
          ],
      ),
      dict(
          testcase_name='large_k',
          k=10,
          n=2,
          use_first_ngram_only=False,
          count_duplicate=True,
          expected_result=[
              ('b b', 2 / 3),
              ('a a', 1 / 3),
              ('c c', 1 / 3),
              ('d a', 1 / 3),
          ],
      ),
      dict(
          testcase_name='large_n',
          k=4,
          n=10,
          use_first_ngram_only=False,
          count_duplicate=True,
          expected_result=[],
      ),
      dict(
          testcase_name='2k_1n',
          k=2,
          n=1,
          use_first_ngram_only=False,
          count_duplicate=True,
          expected_result=[
              ('b', 3 / 3),
              ('a', 2 / 3),
              # ('c', 2 / 3),
              # ('d', 1 / 3),
          ],
      ),
      dict(
          testcase_name='2k_3n',
          k=2,
          n=3,
          use_first_ngram_only=False,
          count_duplicate=True,
          expected_result=[
              ('b b b', 1 / 3),
              ('d a a', 1 / 3),
          ],
      ),
  ])
  def test_topk_word_ngrams(
      self, k, n, use_first_ngram_only, count_duplicate, expected_result
  ):
    texts = [
        'c c',
        'b B b',  # Case-insensitive
        'd a a',
    ]
    result = text.topk_word_ngrams(
        texts=texts,
        k=k,
        n=n,
        use_first_ngram_only=use_first_ngram_only,
        count_duplicate=count_duplicate
    )
    self.assertSequenceAlmostEqual(expected_result, result)

  def test_topk_word_ngrams_empty(self):
    # Randomly choose k and n.
    result = text.topk_word_ngrams(texts=[], k=3, n=2)
    self.assertSequenceAlmostEqual([], result)

  def test_topk_word_ngrams_invalid_kn(self):
    kn_pairs = [(0, 1), (1, 0)]
    for k, n in kn_pairs:
      with self.assertRaisesRegex(
          ValueError, 'k and n must be positive integers.'
      ):
        text.topk_word_ngrams(texts=['a'], k=k, n=n)


class PatternFrequencyTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='not_count_duplicate',
          count_duplicate=False,
          expected_result=[
              ('ab', 1 / 2),
              ('mmm', 0),
              ('xyx', 2 / 2),
          ],
      ),
      dict(
          testcase_name='count_duplicate',
          count_duplicate=True,
          expected_result=[
              ('ab', 2 / 2),
              ('mmm', 0),
              ('xyx', 3 / 2),
          ],
      ),
  ])
  def test_pattern_frequency(self, count_duplicate, expected_result):
    batch = ['ab ab xyx', 'xyxyx']
    result = text.pattern_frequency(
        texts=batch,
        patterns=['ab', 'xyx', 'mmm'],
        count_duplicate=count_duplicate,
    )
    self.assertSequenceAlmostEqual(expected_result, sorted(result))

  def test_pattern_frequency_empty(self):
    result = text.pattern_frequency(
        texts=[], patterns=['ab', 'xyx'], count_duplicate=False
    )
    self.assertSequenceEqual([], result)

  @parameterized.named_parameters([
      dict(
          testcase_name='empty_pattern',
          patterns=[],
          expected_regex='Patterns must not be empty.',
      ),
      dict(
          testcase_name='duplicate_patterns',
          patterns=['a', 'a', 'b'],
          expected_regex='Patterns must be unique:',
      ),
  ])
  def test_pattern_frequency_invalid_patterns(self, patterns, expected_regex):
    with self.assertRaisesRegex(ValueError, expected_regex):
      # `texts` is randomly assigned.
      text.pattern_frequency(texts=['a'], patterns=patterns)


if __name__ == '__main__':
  absltest.main()
