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
from typing import Any

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import utils


FrequencyState = utils.FrequencyState


@dataclasses.dataclass(kw_only=True)
class TopKNGrams(base.MergeableMetric):
  """Text top k n-grams metrics.

  Identify the top k frequent n-grams that appear in the sequence of texts,
  along with their corresponding frequencies. When n-grams share the same
  frequency, the alphabetical order is used for sorting. The resulting order
  determines which n-gram is picked.

  Attributes:
    k:
      Number of most frequent n-grams.
    n:
      Number of grams.
    first_ngrams:
      If `True`, only the first n-grams of each text will be used. Otherwise,
      the unique n-grams from all of the texts will be used.
  """

  k: int
  n: int
  first_ngram: bool = False
  _state: FrequencyState = dataclasses.field(
      default_factory=FrequencyState, init=False
  )

  @property
  def state(self) -> FrequencyState:
    return self._state

  def add(self, texts: Sequence[str|Any]) -> list[tuple[str, float]]:
    ngrams_counter = collections.Counter()
    count = 0
    for text in texts:
      if isinstance(text, str):
        count += 1
        # Remove non-alphabetical and non-space characters
        words = re.sub(r'[^a-zA-Z ]+', '', text).lower().split()
        if self.n <= len(words):
          step = len(words) if self.first_ngram else 1
          ngrans = set([
              ' '.join(words[idx : idx + self.n])
              for idx in range(0, len(words) - self.n + 1, step)
          ])
          ngrams_counter.update(ngrans)
    batch_reault = FrequencyState(counter=ngrams_counter, counte=count)
    self._state += batch_reault

    result = batch_reault.result()[:self.k]
    return result

  def merge(self, other: 'TopKNGrams'):
    self._state += other.state

  def result(self) -> list[tuple[str, float]]:
    return self._state.result()[:self.k]
