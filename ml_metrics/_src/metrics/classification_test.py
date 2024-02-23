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
"""Tests for Classification evaluation metrics API."""

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.aggregates import types
from ml_metrics._src.aggregates import utils
from ml_metrics._src.metrics import classification
import numpy as np

InputType = classification.InputType
ConfusionMatrixMetric = classification.ConfusionMatrixMetric


class ClassificationTest(parameterized.TestCase):

  def test_average_type_sample(self):
    y_pred = [["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]]
    y_true = [["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]]
    np.testing.assert_allclose(
        (11 / 16,),
        classification.ClassificationAggFn(
            (classification.ConfusionMatrixMetric.PRECISION,),
            input_type=InputType.MULTICLASS_MULTIOUTPUT,
            average=types.AverageType.SAMPLES,
        )(y_true, y_pred),
    )
    with self.assertRaisesRegex(
        ValueError, "k_list is not supported for average=SAMPLES"
    ):
      _ = classification.ClassificationAggFn(
          (classification.ConfusionMatrixMetric.PRECISION,),
          input_type=InputType.MULTICLASS_MULTIOUTPUT,
          average=types.AverageType.SAMPLES,
          k_list=[1],
      )(y_true, y_pred)

  def test_default_params(self):
    y_true = [1, 0, 1, 1]
    y_pred = [1, 1, 0, 1]
    np.testing.assert_allclose(
        (2 / 3,),
        classification.ClassificationAggFn(
            (classification.ConfusionMatrixMetric.PRECISION,),
        )(y_true, y_pred),
    )

  def test_multiple_metrics(self):
    y_pred = [["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]]
    y_true = [["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]]
    # expected=classification._ConfusionMatrix(tp=6, tn=12, fp=3, fn=3),
    np.testing.assert_allclose(
        (
            2 / 3,
            2 / 3,
            2 / 3,
            1 / 3,
            1 / 2,
        ),
        classification.ClassificationAggFn(
            (
                classification.ConfusionMatrixMetric.PRECISION,
                classification.ConfusionMatrixMetric.RECALL,
                classification.ConfusionMatrixMetric.F1_SCORE,
                classification.ConfusionMatrixMetric.MISS_RATE,
                classification.ConfusionMatrixMetric.THREAT_SCORE,
            ),
            input_type=InputType.MULTICLASS_MULTIOUTPUT,
            average=types.AverageType.MICRO,
        )(y_true, y_pred),
    )

  @parameterized.named_parameters([
      dict(
          testcase_name="Precision",
          metric_fn=classification.precision,
          expected_no_k_list=2 / 3,
          expected_with_k_list=[5 / 8, 2 / 3],
      ),
      dict(
          testcase_name="ppv",
          metric_fn=classification.ppv,
          expected_no_k_list=2 / 3,
          expected_with_k_list=[5 / 8, 2 / 3],
      ),
      dict(
          testcase_name="recall",
          metric_fn=classification.recall,
          expected_no_k_list=2 / 3,
          expected_with_k_list=[5 / 9, 2 / 3],
      ),
      dict(
          testcase_name="f1_score",
          metric_fn=classification.f1_score,
          expected_no_k_list=2 / 3,
          expected_with_k_list=[10 / 17, 2 / 3],
      ),
      dict(
          testcase_name="accuracy",
          metric_fn=classification.accuracy,
          expected_no_k_list=1,
          expected_with_k_list=[1, 1],
      ),
      dict(
          testcase_name="binary_accuracy",
          metric_fn=classification.binary_accuracy,
          expected_no_k_list=3 / 4,
          expected_with_k_list=[17 / 24, 3 / 4],
      ),
      dict(
          testcase_name="sensitivity",
          metric_fn=classification.sensitivity,
          expected_no_k_list=2 / 3,
          expected_with_k_list=[5 / 9, 2 / 3],
      ),
      dict(
          testcase_name="tpr",
          metric_fn=classification.tpr,
          expected_no_k_list=2 / 3,
          expected_with_k_list=[5 / 9, 2 / 3],
      ),
      dict(
          testcase_name="specificity",
          metric_fn=classification.specificity,
          expected_no_k_list=4 / 5,
          expected_with_k_list=[4 / 5, 4 / 5],
      ),
      dict(
          testcase_name="tnr",
          metric_fn=classification.tnr,
          expected_no_k_list=4 / 5,
          expected_with_k_list=[4 / 5, 4 / 5],
      ),
      dict(
          testcase_name="fall_out",
          metric_fn=classification.fall_out,
          expected_no_k_list=1 / 5,
          expected_with_k_list=[1 / 5, 1 / 5],
      ),
      dict(
          testcase_name="fpr",
          metric_fn=classification.fpr,
          expected_no_k_list=1 / 5,
          expected_with_k_list=[1 / 5, 1 / 5],
      ),
      dict(
          testcase_name="miss_rate",
          metric_fn=classification.miss_rate,
          expected_no_k_list=1 / 3,
          expected_with_k_list=[4 / 9, 1 / 3],
      ),
      dict(
          testcase_name="fnr",
          metric_fn=classification.fnr,
          expected_no_k_list=1 / 3,
          expected_with_k_list=[4 / 9, 1 / 3],
      ),
      dict(
          testcase_name="negative_prediction_value",
          metric_fn=classification.negative_prediction_value,
          expected_no_k_list=4 / 5,
          expected_with_k_list=[3 / 4, 4 / 5],
      ),
      dict(
          testcase_name="nvp",
          metric_fn=classification.nvp,
          expected_no_k_list=4 / 5,
          expected_with_k_list=[3 / 4, 4 / 5],
      ),
      dict(
          testcase_name="false_discovery_rate",
          metric_fn=classification.false_discovery_rate,
          expected_no_k_list=1 / 3,
          expected_with_k_list=[3 / 8, 1 / 3],
      ),
      dict(
          testcase_name="false_omission_rate",
          metric_fn=classification.false_omission_rate,
          expected_no_k_list=1 / 5,
          expected_with_k_list=[1 / 4, 1 / 5],
      ),
      dict(
          testcase_name="threat_score",
          metric_fn=classification.threat_score,
          expected_no_k_list=1 / 2,
          expected_with_k_list=[5 / 12, 1 / 2],
      ),
      dict(
          testcase_name="positive_likelihood_ratio",
          metric_fn=classification.positive_likelihood_ratio,
          expected_no_k_list=10 / 3,
          expected_with_k_list=[25 / 9, 10 / 3],
      ),
      dict(
          testcase_name="negative_likelihood_ratio",
          metric_fn=classification.negative_likelihood_ratio,
          expected_no_k_list=5 / 12,
          expected_with_k_list=[5 / 9, 5 / 12],
      ),
      dict(
          testcase_name="diagnostic_odds_ratio",
          metric_fn=classification.diagnostic_odds_ratio,
          expected_no_k_list=8.0,
          expected_with_k_list=[5, 8],
      ),
      dict(
          testcase_name="positive_predictive_value",
          metric_fn=classification.positive_predictive_value,
          expected_no_k_list=2 / 3,
          expected_with_k_list=[5 / 8, 2 / 3],
      ),
      dict(
          testcase_name="intersection_over_union",
          metric_fn=classification.intersection_over_union,
          expected_no_k_list=1 / 2,
          expected_with_k_list=[5 / 12, 1 / 2],
      ),
      dict(
          testcase_name="prevalence",
          metric_fn=classification.prevalence,
          expected_no_k_list=3 / 8,
          expected_with_k_list=[3 / 8, 3 / 8],
      ),
      dict(
          testcase_name="prevalence_threshold",
          metric_fn=classification.prevalence_threshold,
          expected_no_k_list=(utils.pos_sqrt(30) - 3) / 7,
          expected_with_k_list=[3 / 8, (utils.pos_sqrt(30) - 3) / 7],
      ),
      dict(
          testcase_name="matthews_correlation_coefficient",
          metric_fn=classification.matthews_correlation_coefficient,
          expected_no_k_list=7 / 15,
          expected_with_k_list=[utils.pos_sqrt(2 / 15), 7 / 15],
      ),
      dict(
          testcase_name="informedness",
          metric_fn=classification.informedness,
          expected_no_k_list=7 / 15,
          expected_with_k_list=[16 / 45, 7 / 15],
      ),
      dict(
          testcase_name="markedness",
          metric_fn=classification.markedness,
          expected_no_k_list=7 / 15,
          expected_with_k_list=[3 / 8, 7 / 15],
      ),
      dict(
          testcase_name="balanced_accuracy",
          metric_fn=classification.balanced_accuracy,
          expected_no_k_list=11 / 15,
          expected_with_k_list=[61 / 90, 11 / 15],
      ),
  ])
  def test_individual_metrics(
      self, metric_fn, expected_no_k_list, expected_with_k_list
  ):
    y_pred = [["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]]
    y_true = [["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]]
    actual_no_k_list = metric_fn(
        y_true,
        y_pred,
        input_type=InputType.MULTICLASS_MULTIOUTPUT,
        average=types.AverageType.MICRO,
    )
    metric_doc_details = "\n".join(
        metric_fn.__doc__.split("\n")[1:]
    ).strip()  # ignore the description line for comparison
    self.assertEqual(
        metric_doc_details, classification._METRIC_PYDOC_POSTFIX.strip()
    )
    np.testing.assert_allclose(expected_no_k_list, actual_no_k_list)
    k_list = [1, 2]
    # k=[1 2], tp=[5 6], tn=[12 12], fp=[3 3], fn=[4 3]
    actual_with_k_list = metric_fn(
        y_true,
        y_pred,
        input_type=InputType.MULTICLASS_MULTIOUTPUT,
        average=types.AverageType.MICRO,
        k_list=k_list,
    )
    np.testing.assert_allclose(expected_with_k_list, actual_with_k_list)


if __name__ == "__main__":
  absltest.main()
