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
from ml_metrics.google.tools.signal_registry import registry


# TODO: b/368688941 - Remove this alias once all users are migrated to the new
# module structure.
TopKRetrievalAggFn = retrieval.TopKRetrievalAggFn


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def topk_retrieval_metrics(
    metrics: Sequence[retrieval.RetrievalMetric],
    *,
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
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
  return retrieval.TopKRetrievalAggFn(
      metrics=metrics, k_list=k_list, input_type=input_type
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def precision(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Precision Retrieval metric.

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
  return retrieval.TopKRetrieval(
      metrics=retrieval.RetrievalMetric.PRECISION,
      k_list=k_list,
      input_type=input_type,
  ).as_agg_fn()(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def ppv(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute PPV Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.PPV,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def recall(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Recall Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.RECALL,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def sensitivity(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Sensitivity Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.SENSITIVITY,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def tpr(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute TPR Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.TPR,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def intersection_over_union(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Intersection Over Union Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.INTERSECTION_OVER_UNION,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def positive_predictive_value(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Positive Predictive Value Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.POSITIVE_PREDICTIVE_VALUE,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def f1_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute F1 Score Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.F1_SCORE,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def miss_rate(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Miss Rate Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.MISS_RATE,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def mean_average_precision(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Mean Average Precision Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.MEAN_AVERAGE_PRECISION,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def mean_reciprocal_rank(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Mean Reciprocal Rank Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.MEAN_RECIPROCAL_RANK,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def accuracy(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Accuracy Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.ACCURACY,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def dcg_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute DCG Score Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.DCG_SCORE,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def ndcg_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute NDCG Score Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.NDCG_SCORE,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def fowlkes_mallows_index(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Fowlkes Mallows Index Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.FOWLKES_MALLOWS_INDEX,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def false_discovery_rate(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute False Discovery Rate Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.FALSE_DISCOVERY_RATE,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)


@registry.register_signal(
    signal_modality=registry.SignalModality.OTHER,
    enable_telemetry=False,
)
def threat_score(
    y_true,
    y_pred,
    k_list: list[int] | None = None,
    input_type: types.InputType = types.InputType.MULTICLASS_MULTIOUTPUT,
) -> tuple[float, ...]:
  """Compute Threat Score Retrieval metric.

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
  return retrieval.TopKRetrievalAggFn(
      metrics=retrieval.RetrievalMetric.THREAT_SCORE,
      k_list=k_list,
      input_type=input_type,
  )(y_true, y_pred)
