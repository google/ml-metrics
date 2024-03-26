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

from collections.abc import Callable, Sequence
import dataclasses
import re
from typing import Any

from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import utils


MeanState = utils.MeanState


@dataclasses.dataclass()
class AvgCharCount(base.MergeableMetric):
  """Average character count metric.

  The average character count is the mean number of alphabetical characters in
  the non-missing texts.
  """

  _state: MeanState = dataclasses.field(default_factory=MeanState, init=False)

  @property
  def state(self) -> MeanState:
    return self._state

  def add(self, texts: Sequence[str|Any]) -> float:
    char_count = 0
    count = 0
    for text in texts:
      if isinstance(text, str):
        # Remove non-alphabetical characters
        cleaned_up = re.sub(r'[^a-zA-Z]', '', text)
        char_count += len(cleaned_up)
        count += 1

    batch_state = MeanState(total=char_count, count=count)
    self._state += batch_state
    return batch_state.result()

  def merge(self, other: 'AvgCharCount'):
    self._state += other.state

  def result(self) -> float:
    return self._state.result()


@dataclasses.dataclass()
class AvgWordCount(base.MergeableMetric):
  """Average word count metric.

  In the text, non-alphabetical and non-space characters will be removed,
  resulting in words being separated by spaces. Each contraction, however, will
  be counted as a single word. For instance, "I'm" will be treated as one word.
  The average word count is the mean number of words in the non-missing texts.
  """

  _state: MeanState = dataclasses.field(default_factory=MeanState, init=False)

  @property
  def state(self) -> MeanState:
    return self._state

  def add(self, texts: Sequence[str|None]) -> float:
    word_count = 0
    count = 0
    for text in texts:
      if isinstance(text, str):
        # Remove non-alphabetical and non-space characters
        words = re.sub(r'[^a-zA-Z ]', '', text).split(' ')
        word_count += len(words)
        count += 1

    batch_state = MeanState(total=word_count, count=count)
    self._state += batch_state
    return batch_state.result()

  def merge(self, other: 'AvgWordCount'):
    self._state += other.state

  def result(self) -> float:
    return self._state.result()


@dataclasses.dataclass(kw_only=True)
class Tokenize(base.MergeableMetric):
  """Tokenize based metrics.

  Computes the mean number of tokens in the non-missing text samples.

  Attributes:
    tokenizer:
      A callable that accepts text and returns a sequence of elements.
  """

  tokenizer: Callable[[str], Sequence[Any]]
  _state: MeanState = dataclasses.field(default_factory=MeanState, init=False)

  @property
  def state(self) -> MeanState:
    return self._state

  def add(self, texts: Sequence[str]) -> dict[str, float]:
    token_count = 0
    count = 0
    for text in texts:
      if isinstance(text, str):
        token_count += len(self.tokenizer(text))
        count += 1
    batch_state = MeanState(total=token_count, count=count)
    self._state += batch_state
    return batch_state.result()

  def merge(self, other: 'Tokenize'):
    self._state += other.state

  def result(self) -> float:
    return self._state.result()
