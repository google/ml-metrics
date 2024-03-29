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
from ml_metrics._src.aggregates import text
from absl.testing import absltest


class TextTest(parameterized.TestCase):

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
              # This demonstrated that the alphabetical order is used as a
              # tie-breaker in the frequency state.
              # ('c c', 1 / 3),  # Included when k >= 3
              # ('d a', 1 / 3),  # Included when k >= 4
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
              # This demonstrated that the alphabetical order is used as a
              # tie-breaker in the frequency state.
              # ('c c', 1 / 3),  # Included when k >= 3
              # ('d a', 1 / 3),  # Included when k >= 4
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
              # This demonstrated that the alphabetical order is used as a
              # tie-breaker in the frequency state.
              # ('d a', 1 / 3),  # Included when k >= 3
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
              # This demonstrated that the alphabetical order is used as a
              # tie-breaker in the frequency state.
              # ('c', 2 / 3),  # Included when k >= 3
              # ('d', 1 / 3),  # Included when k >= 4
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
  def test_compute_topkwordngrams(
      self, k, n, use_first_ngram_only, count_duplicate, expected_result
  ):
    batch = [
        'c c',
        'b B b',  # Case-insensitive
        'd a a',
    ]
    metric = text.TopKWordNGrams(
        k=k,
        n=n,
        use_first_ngram_only=use_first_ngram_only,
        count_duplicate=count_duplicate,
    )
    batch_result = metric.add(batch)
    self.assertSequenceAlmostEqual(expected_result, batch_result)
    self.assertSequenceAlmostEqual(expected_result, metric.result())

  def test_compute_topkwordngrams_empty(self):
    metric = text.TopKWordNGrams(k=2, n=2)
    batch_result = metric.add([])
    self.assertSequenceEqual([], batch_result)

  def test_topkwordngrams_add(self):
    metric = text.TopKWordNGrams(k=2, n=2)

    batch_0 = [
        'a b',
        'a b',
        'c d',
    ]
    batch_1 = [
        'a b',
        'b c',
        'b c',
    ]

    batch_0_result = metric.add(batch_0)
    expected_batch_0_result = [
        ('a b', 2 / 3),
        ('c d', 1 / 3),
    ]
    self.assertSequenceAlmostEqual(expected_batch_0_result, batch_0_result)

    batch_1_result = metric.add(batch_1)
    expected_batch_1_result = [
        ('b c', 2 / 3),
        ('a b', 1 / 3),
    ]
    self.assertSequenceAlmostEqual(expected_batch_1_result, batch_1_result)

    expected_metric_result = [
        ('a b', 3 / 6),
        ('b c', 2 / 6),
    ]
    self.assertSequenceAlmostEqual(expected_metric_result, metric.result())

  def test_topkwordngrams_merge(self):
    metric_0 = text.TopKWordNGrams(k=2, n=2)
    metric_1 = text.TopKWordNGrams(k=2, n=2)
    batch_0 = [
        'a b',
        'c d',
    ]
    batch_1 = [
        'b c',
        'b c',
    ]
    metric_0.add(batch_0)
    metric_1.add(batch_1)

    metric_0.merge(metric_1)

    expected_result = [
        ('b c', 2 / 4),
        ('a b', 1 / 4),
    ]
    merged_result = metric_0.result()
    self.assertSequenceAlmostEqual(expected_result, merged_result)

  def test_topkwordngrams_invalid_kn(self):
    with self.assertRaisesRegex(
        ValueError, 'k and n must be positive integers.'
    ):
      text.TopKWordNGrams(k=1, n=0)

    with self.assertRaisesRegex(
        ValueError, 'k and n must be positive integers.'
    ):
      text.TopKWordNGrams(k=0, n=1)


if __name__ == '__main__':
  absltest.main()
