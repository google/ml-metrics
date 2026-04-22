"""Utility for ML metrics APIs."""

import itertools
from typing import Any

from ml_metrics._src.aggregates import classification
from ml_metrics._src.aggregates import types
from ml_metrics.google.tools.signal_registry import registry
from ml_metrics._src.tools.telemetry import telemetry


# TODO: b/499440242 - Remove this message once signals are pushed from this
#   file.
@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    usage_category=telemetry.CATEGORY.METRIC,
)
def verify_input(y_true, y_pred, average, input_type, vocab, pos_label):
  """Verifies the input for ML metrics.

  Validates that the positive label is present in the vocabulary when performing
  binary classification with binary input.

  Args:
    y_true: Iterable of true labels.
    y_pred: Iterable of predicted labels.
    average: The averaging strategy (e.g., `types.AverageType`).
    input_type: The type of input data (e.g., `types.InputType`).
    vocab: An optional dictionary mapping labels to indices.
    pos_label: The label to treat as positive.

  Returns:
    None.

  Raises:
    ValueError: If `average` is binary and `input_type` is binary, and
      `pos_label` is not in the inferred or provided vocabulary.

  Example:
    >>> from ml_metrics._src.aggregates import types
    >>> verify_input(
    ...     y_true=[0, 1],
    ...     y_pred=[1, 0],
    ...     average=types.AverageType.BINARY,
    ...     input_type=types.InputType.BINARY,
    ...     vocab={0: 0, 1: 1},
    ...     pos_label=1
    ... )
  """
  if (
      average == types.AverageType.BINARY
      and input_type == types.InputType.BINARY
  ):
    _validate_pos_label(y_true, y_pred, pos_label, vocab)


def _validate_pos_label(
    y_true, y_pred, pos_label: Any, vocab: dict[str, int] | None
):
  vocab = vocab or classification.get_vocab(
      itertools.chain(y_true, y_pred), False
  )
  labels = list(vocab.keys())
  labels.sort()
  if pos_label not in labels:
    raise ValueError(
        f'Pos label {pos_label} not in labels. Please set a valid pos_label'
        f' from: {labels}'
    )
