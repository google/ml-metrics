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
"""Samplewise scoring metrics for text."""

from collections.abc import Callable, Iterable, Sequence
import re
from typing import Any


def alphabetical_char_count(text: str):
  """Computes the number of alphabetical characters."""

  return len(re.sub(r'[^a-zA-Z]', '', text))


def word_count(text: str):
  """Computes the number of words.

  Computes the number of words within the text. Characters that are not letters
  or spaces are taken out of the text, leaving only spaces between words.
  However, each contraction is counted as only one word. For example, "I'm" is
  treated as one word, "Im".

  Args:
    text:
      Input text.
  Returns:
    Number of words.
  """

  return len(re.sub(r'[^a-zA-Z ]', '', text).split())


def token_count(text: str, tokenizer: Callable[[str], Sequence[Any]]):
  """Computes the number of tokens."""

  return len(tokenizer(text))


def exact_match(sample: str, reference: str) -> bool:
  return sample == reference


def sample_startswith_reference_match(sample: str, reference: str) -> bool:
  return sample.startswith(reference)


def reference_startswith_sample_match(sample: str, reference: str) -> bool:
  return reference.startswith(sample)


def reference_in_sample_match(sample: str, reference: str) -> bool:
  return reference in sample


def sample_in_reference_match(sample: str, reference: str) -> bool:
  return sample in reference


def match(
    sample: str,
    reference: str,
    *,
    matchers: Iterable[Callable[[str, str], bool]],
    normalize_fn: Callable[[str], str] = lambda x: x,
) -> list[bool]:
  if normalize_fn:
    sample = normalize_fn(sample)
    reference = normalize_fn(reference)
  return tuple([matcher(sample, reference) for matcher in matchers])
