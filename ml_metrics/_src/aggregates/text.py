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

import collections
from collections.abc import Sequence
import dataclasses
import re

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import utils
from ml_metrics._src.tools.telemetry import telemetry

FrequencyState = utils.FrequencyState


@dataclasses.dataclass(kw_only=True)
class TopKWordNGrams(base.MergeableMetric, base.HasAsAggFn):
  """Top k word n-grams metrics.

  Identify the top `k` frequent occurring word n-grams with a case-insensitive
  approach. The text will first be cleaned by removing non-alphabetic characters
  and spaces, and then converted to lowercase before computing the top k
  n-grams. When multiple n-grams share the same frequency, alphabetical order
  will be used as a tie-breaker. The result is a list of tuples containing the
  n-gram pattern and its corresponding frequency. The list includes either `k`
  or the number of distinct n-grams tuples, whichever is less.


  Attributes:
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
      first occurrence. Default to `True`.
  """

  k: int
  n: int
  use_first_ngram_only: bool = False
  count_duplicate: bool = True
  _state: FrequencyState = dataclasses.field(
      default_factory=FrequencyState, init=False
  )

  def __post_init__(self):
    telemetry.increment_counter(
        api='ml_metrics', category='metric', reference=self.__class__.__name__
    )
    if self.k <= 0 or self.n <= 0:
      raise ValueError(
          f'k and n must be positive integers but k={self.k} and n={self.n} was'
          ' passed.'
      )

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(
        self.__class__,
        k=self.k,
        n=self.n,
        use_first_ngram_only=self.use_first_ngram_only,
        count_duplicate=self.count_duplicate,
    )

  @property
  def state(self) -> FrequencyState:
    return self._state

  def add(self, texts: Sequence[str]) -> list[tuple[str, float]]:
    ngrams_counter = collections.Counter()
    for text in texts:
      # Remove non-alphabetical and non-space characters
      words = re.sub(r'[^a-zA-Z ]+', '', text).lower().split()
      if self.n <= len(words):
        ngrams = []
        if self.use_first_ngram_only:
          ngrams.append(' '.join(words[:self.n]))
        else:
          for idx in range(len(words) - self.n + 1):
            ngrams.append(' '.join(words[idx : idx + self.n]))
        if not self.count_duplicate:
          ngrams = set(ngrams)
        ngrams_counter.update(ngrams)
    batch_reault = FrequencyState(counter=ngrams_counter, count=len(texts))
    self._state.merge(batch_reault)

    result = batch_reault.result()[:self.k]
    return result

  def merge(self, other: 'TopKWordNGrams'):
    # TODO(b/331796958): Optimize storage consumption
    self._state.merge(other.state)

  def result(self) -> list[tuple[str, float]]:
    return self._state.result()[:self.k]


@dataclasses.dataclass(kw_only=True)
class PatternFrequency(base.MergeableMetric, base.HasAsAggFn):
  """Pattern frequency metric.

  Identify the frequency of occurrence for each pattern found within the given
  texts.


  Attributes:
    patterns:
      Sequence of text patterns.
    count_duplicate:
      If `True`, duplicate pattern within the text are included in the total
      count. Otherwise, the count of a pattern will only consider its first
      occurrence. Default to `False`.
  """

  patterns: Sequence[str]
  count_duplicate: bool = True
  _state: FrequencyState = dataclasses.field(
      default_factory=FrequencyState, init=False
  )

  def __post_init__(self):
    telemetry.increment_counter(
        api='ml_metrics', category='metric', reference=self.__class__.__name__
    )
    if not self.patterns:
      raise ValueError('Patterns must not be empty.')

    if len(set(self.patterns)) != len(self.patterns):
      raise ValueError(f'Patterns must be unique: {self.patterns}')

  def as_agg_fn(self) -> base.AggregateFn:
    return base.as_agg_fn(
        self.__class__,
        patterns=self.patterns,
        count_duplicate=self.count_duplicate,
    )

  @property
  def state(self) -> FrequencyState:
    return self._state

  def add(self, texts: Sequence[str]) -> list[tuple[str, float]]:
    batch_frquency_state = FrequencyState()
    for pattern in self.patterns:
      for text in texts:
        num_matches = 0
        if self.count_duplicate:
          matches = list(
              re.finditer(r'(?=({}))'.format(re.escape(pattern)), text)
          )
          num_matches = len(matches)
        elif text.find(pattern) >= 0:
          num_matches = 1
        batch_frquency_state.counter[pattern] += num_matches
    batch_frquency_state.count = len(texts)
    self._state.merge(batch_frquency_state)
    return batch_frquency_state.result()

  def merge(self, other: 'PatternFrequency'):
    self._state.merge(other.state)

  def result(self) -> list[tuple[str, float]]:
    return self._state.result()
