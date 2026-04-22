"""Topk accuracy metric."""

from ml_metrics._src.aggregates import types
from ml_metrics.google.tools.signal_registry import registry
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    usage_category=telemetry.CATEGORY.SIGNAL,
)
def topk_accurate(
    y_pred: types.NumbersT,
    label: int,
    weights: types.NumbersT = 1.0,
    k: int = 1,
) -> bool:
  """Calculate topk accuracy.

  Args:
    y_pred: Prediction scores with shape [num_classes].
    label: The ground truth label with values in [0, num_classes).
    weights: Weight applied to the prediction scores for computing the top-k
      accurate metric. Default is 1.0.
    k: The top-k predictions to consider.

  Returns:
    True if the label is in the top-k predictions, False otherwise.

  Examples:
    >>> topk_accurate([0.1, 0.2, 0.7], 2, k=1)
    True
    >>> topk_accurate([0.1, 0.2, 0.7], 1, k=2)
    True
    >>> topk_accurate([0.1, 0.2, 0.7], 0, weights=[10.0, 1.0, 1.0], k=1)
    True
  """
  weighted_pred = np.asarray(y_pred) * np.asarray(weights)
  # Get indices of top-k predictions
  topk_predictions = np.argsort(weighted_pred)[-k:]
  return label in topk_predictions
