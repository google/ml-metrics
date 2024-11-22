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
"""Individual text stats metrics."""

from collections.abc import Sequence

from ml_metrics import aggregates
from ml_metrics import chainable
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.aggregates import text
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.signals import text as text_scores


def topk_word_ngrams(
    texts: Sequence[str],
    k: int,
    n: int,
    use_first_ngram_only: bool = False,
    count_duplicate: bool = True,
) -> list[tuple[str, float]]:
  """Top k word n-grams metrics.

  Identify the top `k` frequent occurring word n-grams with a case-insensitive
  approach. The text will first be cleaned by removing non-alphabetic characters
  and spaces, and then converted to lowercase before computing the top k
  n-grams. When multiple n-grams share the same frequency, alphabetical order
  will be used as a tie-breaker. The result is a list of tuples containing the
  n-gram pattern and its corresponding frequency. The list includes either `k`
  or the number of distinct n-grams tuples, whichever is less.

  Args:
    texts:
      Sequence of texts.
    k:
      Number of most frequent word n-grams.
    n:
      Number of grams.
    use_first_ngram_only:
      If `True`, only the first n words of each text will be used to form the
      n-grams and `count_duplicate` will be ignored. Otherwise, all words
      present in each text will be considered for generating the n-grams.
      Default to `False`.
    count_duplicate:
      If `True`, duplicate n-grams within the text are included in the total
      count. Otherwise, the count of a unique N-gram will only consider its
      first occurrence.

  Returns:
    List of tuples of ngram and its frequency of appearance as a pair.
  """

  if k <= 0 or n <= 0:
    raise ValueError(
        f'k and n must be positive integers but k={k} and n={n} was passed.'
    )

  return aggregates.MergeableMetricAggFn(
      lazy_fns.trace(text.TopKWordNGrams)(
          k=k,
          n=n,
          use_first_ngram_only=use_first_ngram_only,
          count_duplicate=count_duplicate,
      )
  )(texts)


def pattern_frequency(
    texts: Sequence[str], patterns: Sequence[str], count_duplicate: bool = True
) -> list[tuple[str, float]]:
  """Pattern frequency metric.

  Identify the frequency of occurrence for each pattern found within the given
  texts.


  Args:
    texts:
      Sequence of texts.
    patterns:
      Sequence of text patterns.
    count_duplicate:
      If `True`, duplicate pattern within the text are included in the total
      count. Otherwise, the count of a pattern will only consider its first
      occurrence. Default to `False`.

  Returns:
    List of tuples of pattern and its frequency of appearance as a pair.
  """

  if not patterns:
    raise ValueError('Patterns must not be empty.')

  if len(set(patterns)) != len(patterns):
    raise ValueError(f'Patterns must be unique: {patterns}')

  return aggregates.MergeableMetricAggFn(
      lazy_fns.trace(text.PatternFrequency)(
          patterns=patterns, count_duplicate=count_duplicate
      )
  )(texts)


def avg_alphabetical_char_count(
    texts: Sequence[str],
) -> rolling_stats.MeanAndVariance:
  """Average alphabetical character count metric."""

  if not list(texts):
    raise ValueError('`texts` must not be empty.')

  batch_scorer_fn = chainable.iterate_fn(text_scores.alphabetical_char_count)
  return aggregates.MergeableMetricAggFn(
      lazy_fns.trace(rolling_stats.MeanAndVariance)(
          batch_score_fn=batch_scorer_fn
      )
  )(texts)
