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

from ml_metrics.google.tools.telemetry import telemetry


def _maybe_tuple(
    reference: str | Sequence[str],
) -> Sequence[str]:
  """Converts reference to a sequence if it is a single string."""
  if isinstance(reference, str):
    return (reference,)
  return reference


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='alphabetical_char_count'
)
def alphabetical_char_count(text: str) -> int:
  """Computes the number of alphabetical characters."""
  return len(re.sub(r'[^a-zA-Z]', '', text))


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='word_count'
)
def word_count(text: str) -> int:
  """Computes the number of words.

  Computes the number of words within the text. Characters that are not letters
  or spaces are taken out of the text, leaving only spaces between words.
  However, each contraction is counted as only one word. For example, "I'm" is
  treated as one word, "Im".

  Args:
    text: Input text.

  Returns:
    Number of words.
  """
  return len(_get_words(text))


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='token_count'
)
def token_count(text: str, tokenizer: Callable[[str], Sequence[Any]]) -> int:
  """Computes the number of tokens."""
  return len(tokenizer(text))


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='token_match_rate'
)
def token_match_rate(
    sample: str, reference: str, tokenizer: Callable[[str], Sequence[Any]]
) -> float:
  """Computes the token match rate between sample and reference."""
  sample_tokens = tokenizer(sample)
  reference_tokens = tokenizer(reference)
  matched = 0
  for t1, t2 in zip(sample_tokens, reference_tokens):
    if t1 == t2:
      matched += 1
  length = max(len(sample_tokens), len(reference_tokens))
  if length == 0:
    return 0
  return matched / length


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='exact_match'
)
def exact_match(sample: str, reference: str | Sequence[str]) -> bool:
  """Computes the exact match between sample and reference."""
  references = _maybe_tuple(reference)
  return any(sample == ref for ref in references)


@telemetry.WithTelemetry(
    api='ml_metrics',
    category='signal',
    reference='sample_startswith_reference_match',
)
def sample_startswith_reference_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the sample starts with reference."""
  references = _maybe_tuple(reference)
  return any(sample.startswith(ref) for ref in references)


@telemetry.WithTelemetry(
    api='ml_metrics',
    category='signal',
    reference='reference_startswith_sample_match',
)
def reference_startswith_sample_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the reference starts with sample."""
  references = _maybe_tuple(reference)
  return any(ref.startswith(sample) for ref in references)


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='reference_in_sample_match'
)
def reference_in_sample_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the reference in sample match."""
  references = _maybe_tuple(reference)
  return any(ref in sample for ref in references)


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='sample_in_reference_match'
)
def sample_in_reference_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the sample in reference match."""
  references = _maybe_tuple(reference)
  return any(sample in ref for ref in references)


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='non_ascii_char_count'
)
def non_ascii_char_count(text: str) -> int:
  """Computes the number of non-ascii characters."""
  return len(re.sub(r'[^\x00-\x7F]+', '', text))


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='is_all_whitespace'
)
def is_all_whitespace(text: str) -> bool:
  r"""Checks if the text is all whitespace.

  Check if string is empty-ish e.g. consisting of whitespace, \n, \t.

  Args:
    text: Input text.

  Returns:
    True if the text is all whitespace.
  """
  return not text.strip()


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='average_word_length'
)
def average_word_length(text: str) -> float:
  """Computes the average word length."""
  words = _get_words(text)
  if not words:
    return 0.0
  return sum(len(word) for word in words) / len(words)


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='average_word_length'
)
def _get_words(text: str) -> list[str]:
  """Returns the words in the text."""
  return re.sub(r'[^a-zA-Z ]', '', text).split()


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='unique_word_count'
)
def unique_word_count(text: str) -> int:
  """Computes the number of unique words."""
  return len(set(_get_words(text)))


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='number_of_characters'
)
def number_of_characters(text: str) -> int:
  """Computes the number of characters."""
  return len(text)


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='percentage_all_caps'
)
def percentage_all_caps(text: str) -> float:
  """Computes the percentage of all caps."""
  words = _get_words(text)
  if not words:
    return 0
  return len([word for word in words if word.isupper()]) / len(words)


@telemetry.WithTelemetry(
    api='ml_metrics',
    category='signal',
    reference='percentage_non_ascii_characters',
)
def percentage_non_ascii_characters(text: str) -> float:
  """Computes the percentage of non-ascii characters."""
  if not number_of_characters(text):
    return 0
  return 1 - (non_ascii_char_count(text) / number_of_characters(text))


@telemetry.WithTelemetry(
    api='ml_metrics', category='signal', reference='type_token_ratio'
)
def type_token_ratio(text: str) -> float:
  """Computes the type token ratio.

  Words with the same letters but different lowercase letters are considered
  different.

  Args:
    text: Input text.

  Returns:
    The ratio of unique words to total words.
  """
  words = _get_words(text)
  if not words:
    return 0

  return unique_word_count(text) / len(words)
