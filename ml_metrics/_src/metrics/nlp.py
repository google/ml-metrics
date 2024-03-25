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
"""Individual NLP based metrics."""

from collections.abc import Sequence

from ml_metrics import aggregates
from ml_metrics._src.aggregates import nlp


def avg_char_count(texts: Sequence[str|None]) -> float:
  """Compute average character count metric.

  The average character count is the mean number of alphabetical characters in
  the non-missing texts.

  Args:
    texts: Sequence of texts.

  Returns:
    Metric value.
  """
  return aggregates.MergeableMetricAggFn(metric=nlp.AvgCharCountMaker())(texts)


def avg_word_count(texts: Sequence[str|None]) -> float:
  """Compute average word count metric.

  In the text, non-alphabetical and non-space characters will be removed,
  resulting in words being separated by spaces. Each contraction, however, will
  be counted as a single word. For instance, "I'm" will be treated as one word.
  The average word count is the mean number of words in the non-missing texts.

  Args:
    texts: Sequence of texts.

  Returns:
    Metric value.
  """
  return aggregates.MergeableMetricAggFn(metric=nlp.AvgWordCountMaker())(texts)
