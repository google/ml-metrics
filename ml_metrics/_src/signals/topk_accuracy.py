"""Topk accuracy metric."""

from ml_metrics._src.aggregates import types
import numpy as np


def topk_accurate(
    y_pred: types.NumbersT,
    label: int,
    k: int,
) -> bool:
  """Calculate topk accuracy.

  Args:
    y_pred: prediction probabilities with shape [num_classes]
    label: the ground truth label with values in [0, num_classes)
    k: the top-k predictions to consider

  Returns:
    True if the label is in the top-k predictions, False otherwise.
  """
  topk_predictions = np.argsort(y_pred)[-k:]  # Get indices of top-k predictions
  return label in topk_predictions
