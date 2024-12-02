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
"""Individual NLP-based metrics."""

from collections.abc import Sequence
import dataclasses
import re

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

  def add(self, texts: Sequence[str|None]) -> float:
    char_count = 0
    non_missing_text_count = 0
    for text in texts:
      if text is not None:
        cleaned_up = re.sub(r'[^a-zA-Z]', '', text)
        char_count += len(cleaned_up)
        non_missing_text_count += 1

    batch_state = MeanState(total=char_count, count=non_missing_text_count)
    self._state += batch_state
    return batch_state.result()

  def merge(self, other: 'AvgCharCount'):
    self._state += other.state

  def result(self) -> float:
    return self._state.result()


@dataclasses.dataclass(kw_only=True)
class AvgCharCountMaker(base.MetricMaker):
  """Average character count metric maker."""

  def make(self):
    return AvgCharCount()


@dataclasses.dataclass(kw_only=True)
class AvgWordCount(base.MergeableMetric):
  """Average word count metric."""

  _state: MeanState = dataclasses.field(default_factory=MeanState, init=False)

  @property
  def state(self) -> MeanState:
    return self._state

  def add(self, texts: Sequence[str|None]) -> float:
    word_count = 0
    non_missing_text_count = 0
    for text in texts:
      if text is not None:
        words = re.sub(r'[^a-zA-Z ]', '', text).split(' ')
        word_count += len(words)
        non_missing_text_count += 1

    batch_state = MeanState(total=word_count, count=non_missing_text_count)
    self._state += batch_state
    return batch_state.result()

  def merge(self, other: 'AvgWordCount'):
    self._state += other.state

  def result(self) -> float:
    return self._state.result()


@dataclasses.dataclass(kw_only=True)
class AvgWordCountMaker(base.MetricMaker):
  """Average word count metric maker."""

  def make(self):
    return AvgWordCount()
