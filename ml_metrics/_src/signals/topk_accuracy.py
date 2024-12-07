"""Topk accuracy metric."""

from ml_metrics._src.aggregates import types
import numpy as np


def topk_accurate(
    y_pred: types.NumbersT,
    label: int,
    weights: types.NumbersT = 1.0,
    k: int = 1,
) -> bool:
  """Calculate topk accuracy.

  Args:
    y_pred: prediction scores with shape [num_classes].
    label: the ground truth label with values in [0, num_classes).
    weights: weight applied to the prediction scores for computing the
      top-k accurate metric. Default is 1.0.
    k: the top-k predictions to consider.

  Returns:
    True if the label is in the top-k predictions, False otherwise.
  """
  weighted_pred = np.asarray(y_pred) * np.asarray(weights)
  # Get indices of top-k predictions
  topk_predictions = np.argsort(weighted_pred)[-k:]
  return label in topk_predictions
