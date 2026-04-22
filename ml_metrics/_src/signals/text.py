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

from ml_metrics.google.tools.signal_registry import registry
from ml_metrics._src.tools.telemetry import telemetry


def _maybe_tuple(
    reference: str | Sequence[str],
) -> Sequence[str]:
  """Converts reference to a sequence if it is a single string."""
  if isinstance(reference, str):
    return (reference,)
  return reference


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def alphabetical_char_count(text: str) -> int:
  """Computes the number of alphabetical characters.

  Args:
    text: The input string.

  Returns:
    The number of alphabetical characters (a-z, A-Z) in the text.

  Examples:
    >>> alphabetical_char_count("Hello World!")
    10
  """
  return len(re.sub(r'[^a-zA-Z]', '', text))


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
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

  Examples:
    >>> word_count("Hello world")
    2
  """
  return len(_get_words(text))


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def token_count(text: str, tokenizer: Callable[[str], Sequence[Any]]) -> int:
  """Computes the number of tokens.

  Args:
    text: Input text.
    tokenizer: A callable that takes a string and returns a sequence of tokens.

  Returns:
    The number of tokens.

  Examples:
    >>> token_count("a b c", lambda x: x.split())
    3
  """
  return len(tokenizer(text))


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def token_match_rate(
    sample: str, reference: str, tokenizer: Callable[[str], Sequence[Any]]
) -> float:
  """Computes the token match rate between sample and reference.

  Args:
    sample: The sample text.
    reference: The reference text.
    tokenizer: A callable that takes a string and returns a sequence of tokens.

  Returns:
    The token match rate (number of matched tokens divided by the maximum
    length of sample or reference tokens).

  Examples:
    >>> token_match_rate("a b", "a c", lambda x: x.split())
    0.5
  """
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


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def exact_match(sample: str, reference: str | Sequence[str]) -> bool:
  """Computes the exact match between sample and reference.

  Args:
    sample: The sample text.
    reference: A string or a sequence of strings to compare against.

  Returns:
    True if the sample matches any of the references.

  Examples:
    >>> exact_match("hello", "hello")
    True
  """
  references = _maybe_tuple(reference)
  return any(sample == ref for ref in references)


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def sample_startswith_reference_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the sample starts with reference.

  Args:
    sample: The sample text.
    reference: A string or a sequence of strings to check against.

  Returns:
    True if the sample starts with any of the references.

  Examples:
    >>> sample_startswith_reference_match("hello world", "hello")
    True
  """
  references = _maybe_tuple(reference)
  return any(sample.startswith(ref) for ref in references)


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def reference_startswith_sample_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the reference starts with sample.

  Args:
    sample: The sample text.
    reference: A string or a sequence of strings to check.

  Returns:
    True if any of the references start with the sample.

  Examples:
    >>> reference_startswith_sample_match("hello", "hello world")
    True
  """
  references = _maybe_tuple(reference)
  return any(ref.startswith(sample) for ref in references)


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def reference_in_sample_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the reference in sample match.

  Args:
    sample: The sample text.
    reference: A string or a sequence of strings to search for.

  Returns:
    True if any of the references are found within the sample.

  Examples:
    >>> reference_in_sample_match("hello world", "world")
    True
  """
  references = _maybe_tuple(reference)
  return any(ref in sample for ref in references)


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def sample_in_reference_match(
    sample: str, reference: str | Sequence[str]
) -> bool:
  """True when the sample in reference match.

  Args:
    sample: The sample text to search for.
    reference: A string or a sequence of strings to search within.

  Returns:
    True if the sample is found within any of the references.

  Examples:
    >>> sample_in_reference_match("world", "hello world")
    True
  """
  references = _maybe_tuple(reference)
  return any(sample in ref for ref in references)


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def non_ascii_char_count(text: str) -> int:
  """Computes the number of ASCII characters.

  Note: Despite the name, this function returns the number of ASCII characters
  because it removes all non-ASCII characters and counts the length of the
  remaining string.

  Args:
    text: Input text.

  Returns:
    The number of ASCII characters in the text.

  Examples:
    >>> non_ascii_char_count("abc")
    3
    >>> non_ascii_char_count("abc©")
    3
  """
  return len(re.sub(r'[^\x00-\x7F]+', '', text))


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def is_all_whitespace(text: str) -> bool:
  r"""Checks if the text is all whitespace.

  Check if string is empty-ish e.g. consisting of whitespace, \n, \t.

  Args:
    text: Input text.

  Returns:
    True if the text is all whitespace.

  Examples:
    >>> is_all_whitespace("   ")
    True
  """
  return not text.strip()


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def average_word_length(text: str) -> float:
  """Computes the average word length.

  Args:
    text: Input text.

  Returns:
    The average length of words in the text.

  Examples:
    >>> average_word_length("abc def")
    3.0
  """
  words = _get_words(text)
  if not words:
    return 0.0
  return sum(len(word) for word in words) / len(words)


def _get_words(text: str) -> list[str]:
  """Returns the words in the text."""
  return re.sub(r'[^a-zA-Z ]', '', text).split()


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def unique_word_count(text: str) -> int:
  """Computes the number of unique words.

  Args:
    text: Input text.

  Returns:
    The number of unique words.

  Examples:
    >>> unique_word_count("hello world hello")
    2
  """
  return len(set(_get_words(text)))


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def number_of_characters(text: str) -> int:
  """Computes the number of characters.

  Args:
    text: Input text.

  Returns:
    The number of characters in the text.

  Examples:
    >>> number_of_characters("abc")
    3
  """
  return len(text)


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def percentage_all_caps(text: str) -> float:
  """Computes the percentage of all caps.

  Args:
    text: Input text.

  Returns:
    The percentage of words that are all capitalized.

  Examples:
    >>> percentage_all_caps("HELLO world")
    0.5
  """
  words = _get_words(text)
  if not words:
    return 0
  return len([word for word in words if word.isupper()]) / len(words)


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def percentage_non_ascii_characters(text: str) -> float:
  """Computes the percentage of non-ascii characters.

  Args:
    text: Input text.

  Returns:
    The percentage of non-ASCII characters in the text.

  Examples:
    >>> percentage_non_ascii_characters("abc")
    0.0
    >>> percentage_non_ascii_characters("abc©")
    0.25
  """
  if not number_of_characters(text):
    return 0
  return 1 - (non_ascii_char_count(text) / number_of_characters(text))


@registry.register_signal(
    signal_modality=registry.SignalModality.TEXT,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def type_token_ratio(text: str) -> float:
  """Computes the type token ratio.

  Words with the same letters but different lowercase letters are considered
  different.

  Args:
    text: Input text.

  Returns:
    The ratio of unique words to total words.

  Examples:
    >>> type_token_ratio("hello world hello")
    0.6666666666666666
  """
  words = _get_words(text)
  if not words:
    return 0

  return unique_word_count(text) / len(words)
