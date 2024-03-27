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
from ml_metrics._src.aggregates import text


def topk_ngrams_first(
    texts: Sequence[str | None], k: int = 5, n: int = 3
) -> list[tuple[str, float]]:
  """Top k first n-grams metrics.

  Identify the top k frequent first n-grams that appear in the sequence of texts
  , along with their corresponding frequencies. When n-grams share the same
  frequency, the alphabetical order is used for sorting. The resulting order
  determines which n-gram is picked.

  Args:
    texts: Sequence of
    k: Number of most frequent n-grams. Default to 5.
    n: Number of grams. Default to 3.

  Returns:
    List of tuples of ngram and its frequency of appearance as a pair.
  """
  return aggregates.MergeableMetricAggFn(
      metric=text.TopKNGrams(k=k, n=n, first_ngram=True)
  )(texts)


def topk_ngrams_all(
    texts: Sequence[str | None], k: int = 5, n: int = 3
) -> list[tuple[str, float]]:
  """Top k n-grams metrics.

  Identify the top k frequent n-grams that appear in the sequence of texts, 
  along with their corresponding frequencies. When n-grams share the same
  frequency, the alphabetical order is used for sorting. The resulting order
  determines which n-gram is picked.

  Args:
    texts: Sequence of
    k: Number of most frequent n-grams. Default to 5.
    n: Number of grams. Default to 3.

  Returns:
    List of tuples of ngram and its frequency of appearance as a pair.
  """
  return aggregates.MergeableMetricAggFn(
      metric=text.TopKNGrams(k=k, n=n, first_ngram=False)
  )(texts)
