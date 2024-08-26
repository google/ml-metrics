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

from collections.abc import Callable, Sequence
import re
from typing import Any


def _convert_to_sequence(
    reference: str | Sequence[str],
) -> Sequence[str]:
  """Converts reference to a sequence if it is a single string."""
  if isinstance(reference, str):
    return (reference,)
  return reference


def alphabetical_char_count(text: str) -> int:
  """Computes the number of alphabetical characters."""
  return len(re.sub(r'[^a-zA-Z]', '', text))


def word_count(text: str) -> int:
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


def token_count(text: str, tokenizer: Callable[[str], Sequence[Any]]) -> int:
  """Computes the number of tokens."""
  return len(tokenizer(text))


def exact_match(sample: str, reference: str | Sequence[str]) -> bool:
  """Computes the exact match between sample and reference."""
  references = _convert_to_sequence(reference)
  return any(sample == ref for ref in references)


def sample_startswith_reference_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the sample starts with reference."""
  references = _convert_to_sequence(reference)
  return any(sample.startswith(ref) for ref in references)


def reference_startswith_sample_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the reference starts with sample."""
  references = _convert_to_sequence(reference)
  return any(ref.startswith(sample) for ref in references)


def reference_in_sample_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the reference in sample match."""
  references = _convert_to_sequence(reference)
  return any(ref in sample for ref in references)


def sample_in_reference_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the sample in reference match."""
  references = _convert_to_sequence(reference)
  return any(sample in ref for ref in references)


def non_ascii_char_count(text: str) -> int:
  """Computes the number of non-ascii characters."""
  return len(re.sub(r'[^\x00-\x7F]+', '', text))


def is_all_whitespace(text: str) -> bool:
  r"""Checks if the text is all whitespace.

  Check if string is empty-ish e.g. consisting of whitespace, \n, \t.

  Args:
    text: Input text.

  Returns:
    True if the text is all whitespace.
  """
  return not text.strip()


def average_word_length(text: str) -> float:
  """Computes the average word length."""
  words = re.sub(r'[^a-zA-Z ]', '', text).split()
  if not words:
    return 0.0
  return sum(len(word) for word in words) / len(words)
