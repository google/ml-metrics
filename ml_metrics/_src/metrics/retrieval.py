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
"""Individual Retrieval based metrics."""

from collections.abc import Sequence
from ml_metrics._src.aggregates import retrieval
from ml_metrics._src.aggregates import types

InputType = types.InputType
RetrievalMetric = retrieval.RetrievalMetric
TopKRetrievalAggFn = retrieval.TopKRetrievalAggFn

_METRIC_PYDOC_POSTFIX = """

  Args:
    y_true: array of sample's true labels
    y_pred: array of sample's label predictions
    k_list: k_list is only applicable for average_type != Samples and
      multiclass/multioutput input types. It is a list of topk each of which
      slices y_pred by y_pred[:topk] assuming the predictions are sorted in
      descending order. Default 'None' means consider all outputs in the
      prediction.
        input_type: one input type from types.InputType

  Returns:
    Tuple with metric value(s)
"""


def compute_metrics(
    metrics: Sequence[RetrievalMetric],
    *,
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[tuple[float, ...], ...]:
  """Compute multiple metrics together for better efficiency.

  Args:
    metrics: List of CFM metrics
    y_true: array of sample's true labels
    y_pred: array of sample's label predictions
    k_list: k_list is only applicable for average_type != Samples and
      multiclass/multioutput input types. It is a list of topk each of which
      slices y_pred by y_pred[:topk] assuming the predictions are sorted in
      descending order. Default 'None' means consider all outputs in the
      prediction.
    input_type: one input type from types.InputType

  Returns:
    Tuple containing the evaluation metric values. in the corresponding order of
      given metric names in metrics list.
  """
  return TopKRetrievalAggFn(
      metrics=metrics, k_list=k_list, input_type=input_type
  )(y_true, y_pred)


def precision(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Precision Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.PRECISION,), k_list=k_list, input_type=input_type
  )(y_true, y_pred)[0]
precision.__doc__ += _METRIC_PYDOC_POSTFIX


def ppv(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute PPV Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.PPV,), k_list=k_list, input_type=input_type
  )(y_true, y_pred)[0]
ppv.__doc__ += _METRIC_PYDOC_POSTFIX


def recall(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Recall Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.RECALL,), k_list=k_list, input_type=input_type
  )(y_true, y_pred)[0]
recall.__doc__ += _METRIC_PYDOC_POSTFIX


def sensitivity(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Sensitivity Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.SENSITIVITY,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
sensitivity.__doc__ += _METRIC_PYDOC_POSTFIX


def tpr(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute TPR Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.TPR,), k_list=k_list, input_type=input_type
  )(y_true, y_pred)[0]
tpr.__doc__ += _METRIC_PYDOC_POSTFIX


def intersection_over_union(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Intersection Over Union Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.INTERSECTION_OVER_UNION,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
intersection_over_union.__doc__ += _METRIC_PYDOC_POSTFIX


def positive_predictive_value(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Positive Predictive Value Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.POSITIVE_PREDICTIVE_VALUE,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
positive_predictive_value.__doc__ += _METRIC_PYDOC_POSTFIX


def f1_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute F1 Score Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.F1_SCORE,), k_list=k_list, input_type=input_type
  )(y_true, y_pred)[0]
f1_score.__doc__ += _METRIC_PYDOC_POSTFIX


def miss_rate(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Miss Rate Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.MISS_RATE,), k_list=k_list, input_type=input_type
  )(y_true, y_pred)[0]
miss_rate.__doc__ += _METRIC_PYDOC_POSTFIX


def mean_average_precision(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Mean Average Precision Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.MEAN_AVERAGE_PRECISION,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
mean_average_precision.__doc__ += _METRIC_PYDOC_POSTFIX


def mean_reciprocal_rank(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Mean Reciprocal Rank Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.MEAN_RECIPROCAL_RANK,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
mean_reciprocal_rank.__doc__ += _METRIC_PYDOC_POSTFIX


def accuracy(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Accuracy Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.ACCURACY,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
accuracy.__doc__ += _METRIC_PYDOC_POSTFIX


def dcg_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute DCG Score Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.DCG_SCORE,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
dcg_score.__doc__ += _METRIC_PYDOC_POSTFIX


def ndcg_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute NDCG Score Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.NDCG_SCORE,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
ndcg_score.__doc__ += _METRIC_PYDOC_POSTFIX


def fowlkes_mallows_index(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Fowlkes Mallows Index Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.FOWLKES_MALLOWS_INDEX,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
fowlkes_mallows_index.__doc__ += _METRIC_PYDOC_POSTFIX


def false_discovery_rate(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute False Discovery Rate Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.FALSE_DISCOVERY_RATE,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
false_discovery_rate.__doc__ += _METRIC_PYDOC_POSTFIX


def threat_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: InputType = InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Threat Score Retrieval metric."""
  return TopKRetrievalAggFn(
      metrics=(RetrievalMetric.THREAT_SCORE,),
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)[0]
threat_score.__doc__ += _METRIC_PYDOC_POSTFIX
