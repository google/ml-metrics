"""Code to compute CG Score.

The Complexity Gap Score (CG Score) quantifies the "influence of individual
instances." CG scores identify observations with large/small influence on
downstream classification performance, potentially flagging label noise issues.
The current implementation works for binary labels only.
More information about the stochastic method implemented here can be found at
https://arxiv.org/pdf/2301.00930.pdf (Section A.2).
"""

from ml_metrics._src.aggregates import types
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np


@telemetry.WithTelemetry('ml_metrics', 'signals', 'complexity_gap_score')
def complexity_gap_score(
    labels: types.NumbersT,
    embeddings: types.NumbersT,
    *,
    num_repetitions: int = 1,
    class_balance_ratio: float = 1.0,
    random_seed: int = 0,
) -> types.NumbersT:
  """Calculates the Complexity Gap (CG) score for identifying influential instances.

  Args:
      labels: Labels in binary vector representations.
      embeddings: Embeddings in vector representations.
      num_repetitions: Number of times to repeat the CG score calculation.
      class_balance_ratio: Ratio for balancing classes during calculation (e.g.,
        1.0 for perfect balance).
      random_seed: Random seed for reproducibility.

  Returns:
      A NumPy array containing the CG score for each data point.
  """
  if (l := len(np.unique(labels))) > 2:
    raise ValueError(f'CG score only works for binary labels, got {l} labels.')

  cg_scores = np.zeros(len(labels))

  if l == 1:
    return cg_scores

  data_by_label = _group_data_by_label(embeddings, labels)
  if random_seed:
    np.random.seed(random_seed)

  for _ in range(num_repetitions):
    for label, data in data_by_label.items():
      data = np.array(data['data'])
      other_label = _get_other_label(data_by_label, label)
      other_data = np.array(data_by_label[other_label]['data'])
      balanced_data = _balance_dataset(data, other_data, class_balance_ratio)
      vi_scores = _calculate_influence_scores(balanced_data, data.shape[0])
      cg_scores[data_by_label[label]['indices']] += vi_scores

  return cg_scores / np.maximum(num_repetitions, 1)


def _group_data_by_label(embeddings: np.ndarray, labels: np.ndarray):
  data_by_label = {}
  for i, (embedding, label) in enumerate(zip(embeddings, labels, strict=True)):
    data_by_label.setdefault(label, {'data': [], 'indices': []})
    data_float = embedding.astype(np.float64)
    data_by_label[label]['data'].append(
        data_float / np.linalg.norm(data_float, axis=-1, keepdims=True)
    )
    data_by_label[label]['indices'].append(i)
  return data_by_label


def _get_other_label(
    data_by_label: dict[int, object], current_label: int
) -> int:
  return [label for label in data_by_label if label != current_label][0]


def _balance_dataset(
    data: np.ndarray, other_data: np.ndarray, ratio: float
) -> np.ndarray:
  max_size = min(int(len(data) * ratio), len(other_data))
  selected_indices = np.random.choice(len(other_data), max_size, replace=False)
  balanced_other_data = other_data[selected_indices]
  return np.concatenate((data, balanced_other_data))


def _calculate_influence_scores(
    data: np.ndarray, data_size: int
) -> types.NumbersT:
  """Computes Complexity Gap "influence" scores for each data point."""

  reformatted_data = data
  y = np.concatenate([np.ones(data_size), -np.ones(data.shape[0] - data_size)])
  hermitian_inner = reformatted_data @ np.transpose(reformatted_data)
  hermitian = (hermitian_inner * (np.pi - np.arccos(hermitian_inner))) / (
      2 * np.pi
  )
  np.fill_diagonal(hermitian, 0.5)
  hermitian[np.isnan(hermitian)] = np.nextafter(np.float64(0), np.float64(1))

  inv_hermitian = np.linalg.pinv(hermitian, hermitian=True)
  original_error = y @ (inv_hermitian @ y)

  influence_scores = np.zeros(data_size)
  for k in range(data_size):
    without_col = np.delete(inv_hermitian, k, axis=1)
    without_k = np.delete(without_col, k, axis=0)
    row_expanded = np.expand_dims(without_col[k, :], 0)
    with_k = inv_hermitian[k, k]

    inv_hermitian_except_k = (
        without_k - (row_expanded.transpose() @ row_expanded) / with_k
    )
    y_except_k = np.delete(y, k, axis=0)
    influence_scores[k] = original_error - y_except_k @ (
        inv_hermitian_except_k @ y_except_k
    )

  return influence_scores
