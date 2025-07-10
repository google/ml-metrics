"""Utility for ML metrics APIs."""

import itertools
from typing import Any

from ml_metrics._src.aggregates import classification
from ml_metrics._src.aggregates import types
from ml_metrics._src.tools.telemetry import telemetry


@telemetry.function_monitor(
    api='ml_metrics', category=telemetry.CATEGORY.METRIC
)
def verify_input(y_true, y_pred, average, input_type, vocab, pos_label):
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
