# Copyright 2023 Google LLC
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

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.aggregates import classification
from ml_metrics._src.aggregates import types
from ml_metrics._src.aggregates import utils
import numpy as np


InputType = types.InputType
AverageType = types.AverageType
ConfusionMatrixMetric = classification.ConfusionMatrixMetric
_ImplementedDerivedConfusionMatrixMetrics = [
    "precision",
    "ppv",
    "recall",
    "f1_score",
    "accuracy",
    "binary_accuracy",
    "sensitivity",
    "tpr",
    "specificity",
    "tnr",
    "fall_out",
    "fpr",
    "miss_rate",
    "fnr",
    "negative_prediction_value",
    "nvp",
    "false_discovery_rate",
    "false_omission_rate",
    "threat_score",
    "positive_likelihood_ratio",
    "negative_likelihood_ratio",
    "diagnostic_odds_ratio",
    "positive_predictive_value",
    "intersection_over_union",
    "prevalence",
    "prevalence_threshold",
    "matthews_correlation_coefficient",
    "informedness",
    "markedness",
    "balanced_accuracy",
]


class ClassificationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="binary_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="binary",
          expected=classification._ConfusionMatrix(tp=2, tn=2, fp=1, fn=2),
      ),
      dict(
          testcase_name="multiclass_indicator_binary_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="binary",
          expected=classification._ConfusionMatrix(tp=2, tn=2, fp=1, fn=2),
      ),
      dict(
          testcase_name="binary_string_binary_average",
          y_pred=["Y", "N", "Y", "N", "Y", "N", "N"],
          y_true=["Y", "Y", "N", "N", "Y", "N", "Y"],
          input_type="binary",
          pos_label="Y",
          average="binary",
          expected=classification._ConfusionMatrix(tp=2, tn=2, fp=1, fn=2),
      ),
      dict(
          testcase_name="binary_micro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          # 2 CFM are created and sum of each CFM's tp, tn, fp, fn is computed.
          average="micro",
          expected=classification._ConfusionMatrix(tp=4, tn=4, fp=3, fn=3),
      ),
      dict(
          testcase_name="multiclass_indicator_micro_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="micro",
          expected=classification._ConfusionMatrix(tp=4, tn=4, fp=3, fn=3),
      ),
      dict(
          testcase_name="multiclass_micro_average",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          # 3 CFM are created and sum of each CFM's tp, tn, fp, fn is computed.
          average="micro",
          expected=classification._ConfusionMatrix(tp=5, tn=13, fp=3, fn=3),
      ),
      # 3 confusion matrices are:
      # 'n':
      #  [[2 2]
      #  [2 2]]
      #
      # 'u':
      #  [[7 0]
      #  [0 1]]
      #
      # 'y':
      #  [[3 1]
      #  [1 3]]
      dict(
          testcase_name="multiclass_multioutput_micro_average",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          average="micro",
          expected=classification._ConfusionMatrix(tp=6, tn=12, fp=3, fn=3),
      ),
      dict(
          testcase_name="binary_macro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="macro",
          expected=classification._ConfusionMatrix(
              tp=[2, 2],
              tn=[2, 2],
              fp=[1, 2],
              fn=[2, 1],
          ),
      ),
      dict(
          testcase_name="multiclass_indicator_macro_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="macro",
          expected=classification._ConfusionMatrix(
              tp=[2, 2],
              tn=[2, 2],
              fp=[1, 2],
              fn=[2, 1],
          ),
      ),
      dict(
          testcase_name="multiclass_macro_average",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          # A vocab is required to align class index for macro.
          vocab={"y": 0, "n": 1, "u": 2},
          average="macro",
          expected=classification._ConfusionMatrix(
              tp=[2, 2, 1],
              tn=[3, 3, 7],
              fp=[1, 2, 0],
              fn=[2, 1, 0],
          ),
      ),
      dict(
          testcase_name="multiclass_multioutput_macro_average",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          # A vocab is required to align class index for macro.
          vocab={"y": 0, "n": 1, "u": 2},
          average="macro",
          expected=classification._ConfusionMatrix(
              tp=[3, 2, 1],
              tn=[3, 2, 7],
              fp=[1, 2, 0],
              fn=[1, 2, 0],
          ),
      ),
  ])
  def test_confusion_matrix_callable(
      self,
      y_true,
      y_pred,
      input_type,
      average,
      expected,
      vocab=None,
      pos_label=1,
  ):
    confusion_matrix = classification.ConfusionMatrixAggFn(
        input_type=input_type,
        average=average,
        vocab=vocab,
        pos_label=pos_label,
    )
    self.assertEqual((expected,), confusion_matrix(y_true, y_pred))

  @parameterized.named_parameters([
      dict(
          testcase_name="multiclass_micro_average",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          average="micro",
          k_list=[1, 2],
          # Note: since there are three classes ('y', 'n', 'u'), the negatives
          # and falses count the classes that are absent in y_pred or y_true.
          # TP: The classes appear in both y_pred and y_true:
          #     [1, 0, 0, 1, 1, 1, 0, 1] = 5
          # TN: The classes absent in both y_pred and y_true:
          #     [2, 1, 1, 2, 2, 2, 1, 2] = 13
          # FP: The classes appear in y_pred but not in y_true:
          #     [0, 1, 1, 0, 0, 0, 1, 0] = 3
          # FN: The classes absent in y_pred but in y_true:
          #     [0, 1, 1, 0, 0, 0, 1, 0] = 3
          expected=classification._TopKConfusionMatrix(
              k=[1, 2],
              tp=[5, 5],
              tn=[13, 13],
              fp=[3, 3],
              fn=[3, 3],
          ),
      ),
      dict(
          testcase_name="multiclass_multioutput_micro_average",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          average="micro",
          k_list=[1, 2],
          # Note: since there are three classes ('y', 'n', 'u'), the negatives
          # and falses count the classes that ae absent in y_pred or y_true.
          # TP: The classes appear in both y_pred and y_true:
          #     K = 1: [1, 0, 0, 1, 1, 1, 0, 1] = 5
          #     K = 2: [1, 1, 0, 1, 1, 1, 0, 1] = 6
          # TN: The classes absent in both y_pred and y_true
          #     K = 1: [2, 1, 1, 2, 2, 2, 1, 2] = 12
          #     K = 2: [2, 1, 1, 2, 2, 2, 1, 2] = 12
          # FP: The classes appear in y_pred but not in y_true:
          #     K = 1: [0, 1, 1, 0, 0, 0, 1, 0] = 3
          #     K = 2: [0, 1, 1, 0, 0, 0, 1, 0] = 3
          # FN: The classes absent in y_pred but in y_true:
          #     K = 1: [0, 1, 1, 0, 1, 0, 1, 0] = 4
          #     K = 2: [0, 1, 1, 0, 0, 0, 1, 0] = 3
          expected=classification._TopKConfusionMatrix(
              k=[1, 2],
              tp=[5, 6],
              tn=[12, 12],
              fp=[3, 3],
              fn=[4, 3],
          ),
      ),
      dict(
          testcase_name="multiclass_macro_average",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          # A vocab is required to align class index for macro.
          vocab={"y": 0, "n": 1, "u": 2},
          average="macro",
          k_list=[1, 2],
          # The macro average computes a per-class confusion matrix. Since there
          # 3 classes ('y', 'n', 'u'), we have a K x 3 2D array as a result.
          # TP: The classes appear in both y_pred and y_true:
          #     K = 1: [2, 2, 1] in the order of 'y', 'n', 'u'.
          #     K = 2: Same since there is only one output.
          # TN: The classes absent in both y_pred and y_true
          #     K = 1: [3, 3, 7] in the order of 'y', 'n', 'u'.
          #     K = 2: Same since there is only one output.
          # FP: The classes appear in y_pred but not in y_true:
          #     K = 1: [1, 2, 0] in the order of 'y', 'n', 'u'.
          #     K = 2: Same since there is only one output.
          # FN: The classes absent in y_pred but in y_true:
          #     K = 1: [2, 1, 0] in the order of 'y', 'n', 'u'.
          #     K = 2: Same since there is only one output.
          expected=classification._TopKConfusionMatrix(
              k=[1, 2],
              tp=[[2, 2, 1], [2, 2, 1]],
              tn=[[3, 3, 7], [3, 3, 7]],
              fp=[[1, 2, 0], [1, 2, 0]],
              fn=[[2, 1, 0], [2, 1, 0]],
          ),
      ),
      dict(
          testcase_name="multiclass_multioutput_macro_average",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          # A vocab is required to align class index for macro.
          vocab={"y": 0, "n": 1, "u": 2},
          average="macro",
          k_list=[1, 2],
          # The macro average compues a per-class confusion matrix. Since there
          # 3 classes ('y', 'n', 'u'), we have a K x 3 2D array as a result.
          # TP: The classes appear in both y_pred and y_true:
          #     K = 1: [2, 2, 1] in the order of 'y', 'n', 'u'.
          #     K = 2: [3, 2, 1] in the order of 'y', 'n', 'u'.
          # TN: The classes absent in both y_pred and y_true
          #     K = 1: [3, 2, 7] in the order of 'y', 'n', 'u'.
          #     K = 2: [3, 2, 7] in the order of 'y', 'n', 'u'.
          # FP: The classes appear in y_pred but not in y_true:
          #     K = 1: [1, 2, 0] in the order of 'y', 'n', 'u'.
          #     K = 2: [1, 2, 0] in the order of 'y', 'n', 'u'.
          # FN: The classes absent in y_pred but in y_true:
          #     K = 1: [2, 2, 0] in the order of 'y', 'n', 'u'.
          #     K = 2: [1, 2, 0] in the order of 'y', 'n', 'u'.
          expected=classification._TopKConfusionMatrix(
              k=[1, 2],
              tp=[[2, 2, 1], [3, 2, 1]],
              tn=[[3, 2, 7], [3, 2, 7]],
              fp=[[1, 2, 0], [1, 2, 0]],
              fn=[[2, 2, 0], [1, 2, 0]],
          ),
      ),
  ])
  def test_topk_confusion_matrix_callable(
      self,
      y_true,
      y_pred,
      input_type,
      k_list,
      average,
      expected,
      vocab=None,
      pos_label=1,
  ):
    confusion_matrix = classification.TopKConfusionMatrixAggFn(
        input_type=input_type,
        average=average,
        k_list=k_list,
        vocab=vocab,
        pos_label=pos_label,
    )
    self.assertEqual((expected,), confusion_matrix(y_true, y_pred))

  @parameterized.named_parameters([
      # tp=2, tn=2, fp=1, fn=2, p = 3, t = 4
      dict(
          testcase_name="binary_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="binary",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          expected=(
              2 / 3,  # precision
              2 / 3,  # ppv
              2 / 4,  # recall
              4 / 7,  # f1_score
              1,  # accuracy
              4 / 7,  # binary_accuracy
              2 / 4,  # sensitivity
              2 / 4,  # tpr
              2 / 3,  # specificity
              2 / 3,  # tnr
              1 / 3,  # fall_out
              1 / 3,  # fpr
              2 / 4,  # miss_rate
              2 / 4,  # fnr
              2 / 4,  # negative_prediction_value
              2 / 4,  # nvp
              1 / 3,  # false_discovery_rate
              2 / 4,  # false_omission_rate
              2 / 5,  # threat_score
              3 / 2,  # positive_likelihood_ratio
              3 / 4,  # negative_likelihood_ratio
              4 / 2,  # diagnostic_odds_ratio
              2 / 3,  # positive_predictive_value
              2 / 5,  # intersection_over_union
              4 / 7,  # prevalence
              utils.pos_sqrt(6) - 2,  # prevalence_threshold
              1 / 6,  # matthews_correlation_coefficient
              1 / 6,  # informedness
              1 / 6,  # markedness
              7 / 12,  # balanced_accuracy
          ),
      ),
      # tp=[2 2], tn=[2 2], fp=[1 2], fn=[2 1], p = [3 4], t = [4 3]
      dict(
          testcase_name="macro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="macro",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          expected=(
              (2 / 3 + 2 / 4) / 2,  # precision
              (2 / 3 + 2 / 4) / 2,  # ppv
              (2 / 4 + 2 / 3) / 2,  # recall
              4 / 7,  # f1_score
              1,  # accuracy
              (4 / 7 + 4 / 7) / 2,  # binary_accuracy
              (2 / 4 + 2 / 3) / 2,  # sensitivity
              (2 / 4 + 2 / 3) / 2,  # tpr
              (2 / 3 + 2 / 4) / 2,  # specificity
              (2 / 3 + 2 / 4) / 2,  # tnr
              (1 / 3 + 2 / 4) / 2,  # fall_out
              (1 / 3 + 2 / 4) / 2,  # fpr
              (2 / 4 + 1 / 3) / 2,  # miss_rate
              (2 / 4 + 1 / 3) / 2,  # fnr
              (2 / 4 + 2 / 3) / 2,  # negative_prediction_value
              (2 / 4 + 2 / 3) / 2,  # nvp
              (1 / 3 + 2 / 4) / 2,  # false_discovery_rate
              (2 / 4 + 1 / 3) / 2,  # false_omission_rate
              (2 / 5 + 2 / 5) / 2,  # threat_score
              (6 / 4 + 4 / 3) / 2,  # positive_likelihood_ratio
              (3 / 4 + 4 / 6) / 2,  # negative_likelihood_ratio
              (6 / 3 + 6 / 3) / 2,  # diagnostic_odds_ratio
              (2 / 3 + 2 / 4) / 2,  # positive_predictive_value
              (2 / 5 + 2 / 5) / 2,  # intersection_over_union
              (4 / 7 + 3 / 7) / 2,  # prevalence
              ((utils.pos_sqrt(6) - 2) + (2 * utils.pos_sqrt(3) - 3))
              / 2,  # prevalence_threshold
              (1 / 6 + 1 / 6) / 2,  # matthews_correlation_coefficient
              (1 / 6 + 1 / 6) / 2,  # informedness
              (1 / 6 + 1 / 6) / 2,  # markedness
              (7 / 12 + 7 / 12) / 2,  # balanced_accuracy
          ),
      ),
      # tp=2, tn=2, fp=1, fn=2, p = 3, t = 4
      dict(
          testcase_name="multiclass_indicator_binary_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="binary",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          expected=(
              2 / 3,  # precision
              2 / 3,  # ppv
              2 / 4,  # recall
              4 / 7,  # f1_score
              1,  # accuracy
              4 / 7,  # binary_accuracy
              2 / 4,  # sensitivity
              2 / 4,  # tpr
              2 / 3,  # specificity
              2 / 3,  # tnr
              1 / 3,  # fall_out
              1 / 3,  # fpr
              2 / 4,  # miss_rate
              2 / 4,  # fnr
              2 / 4,  # negative_prediction_value
              2 / 4,  # nvp
              1 / 3,  # false_discovery_rate
              2 / 4,  # false_omission_rate
              2 / 5,  # threat_score
              3 / 2,  # positive_likelihood_ratio
              3 / 4,  # negative_likelihood_ratio
              4 / 2,  # diagnostic_odds_ratio
              2 / 3,  # positive_predictive_value
              2 / 5,  # intersection_over_union
              4 / 7,  # prevalence
              utils.pos_sqrt(6) - 2,  # prevalence_threshold
              1 / 6,  # matthews_correlation_coefficient
              1 / 6,  # informedness
              1 / 6,  # markedness
              7 / 12,  # balanced_accuracy
          ),
      ),
      # tp=2, tn=2, fp=1, fn=2, p = 3, t = 4
      dict(
          testcase_name="binary_string_binary_average",
          y_pred=["Y", "N", "Y", "N", "Y", "N", "N"],
          y_true=["Y", "Y", "N", "N", "Y", "N", "Y"],
          input_type="binary",
          pos_label="Y",
          average="binary",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          expected=(
              2 / 3,  # precision
              2 / 3,  # ppv
              2 / 4,  # recall
              4 / 7,  # f1_score
              1,  # accuracy
              4 / 7,  # binary_accuracy
              2 / 4,  # sensitivity
              2 / 4,  # tpr
              2 / 3,  # specificity
              2 / 3,  # tnr
              1 / 3,  # fall_out
              1 / 3,  # fpr
              2 / 4,  # miss_rate
              2 / 4,  # fnr
              2 / 4,  # negative_prediction_value
              2 / 4,  # nvp
              1 / 3,  # false_discovery_rate
              2 / 4,  # false_omission_rate
              2 / 5,  # threat_score
              3 / 2,  # positive_likelihood_ratio
              3 / 4,  # negative_likelihood_ratio
              4 / 2,  # diagnostic_odds_ratio
              2 / 3,  # positive_predictive_value
              2 / 5,  # intersection_over_union
              4 / 7,  # prevalence
              utils.pos_sqrt(6) - 2,  # prevalence_threshold
              1 / 6,  # matthews_correlation_coefficient
              1 / 6,  # informedness
              1 / 6,  # markedness
              7 / 12,  # balanced_accuracy
          ),
      ),
  ])
  def test_confusion_matrix_metric(
      self,
      y_true,
      y_pred,
      input_type,
      average,
      expected,
      metrics,
      vocab=None,
      pos_label=1,
  ):
    confusion_matrix = classification.ConfusionMatrixAggFn(
        input_type=input_type,
        average=average,
        vocab=vocab,
        pos_label=pos_label,
        metrics=metrics,
    )
    np.testing.assert_allclose(
        expected, confusion_matrix(y_true, y_pred), atol=1e-6
    )

  def test_confusion_matrix_metric_invalid_average_type(self):
    confusion_matrix = classification.ConfusionMatrixAggFn(
        metrics=(ConfusionMatrixMetric.PRECISION,),
        average=AverageType.WEIGHTED,
    )
    with self.assertRaisesRegex(
        NotImplementedError, '"weighted" average is not supported'
    ):
      confusion_matrix([0, 1, 0], [1, 0, 0])

  def test_topk_confusion_matrix_invalid_input_type(self):
    with self.assertRaisesRegex(ValueError, '"binary" input is not supported'):
      classification.TopKConfusionMatrixAggFn(input_type=InputType.BINARY)

  def test_confusion_matrix_metric_invalid_metric(self):
    confusion_matrix = classification.ConfusionMatrixAggFn(
        metrics=(ConfusionMatrixMetric.MEAN_AVERAGE_PRECISION,),
    )
    with self.assertRaisesRegex(NotImplementedError, "metric is not supported"):
      confusion_matrix([0, 1, 0], [1, 0, 0])

  @parameterized.named_parameters([
      dict(
          testcase_name="multiclass_indicator",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]],
          input_type="multiclass-indicator",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          # precision = mean([1, 0, 0, 1, 1, 0]) = 3 / 6 = 0.5
          # recall = mean([1, 0, 0, 1, 1, 0]) = 3 / 6 = 0.5
          # f1_score = mean(2 * precision * recall / (precision + recall)) = 0.5
          # accuracy = mean([1, 0, 0, 1, 1, 0]) = 0.5
          expected=(
              0.5,  # 'precision': MeanState(total=3.0, count=6)
              0.5,  # 'ppv': MeanState(total=3.0, count=6)
              0.5,  # 'recall': MeanState(total=3.0, count=6)
              0.5,  # 'f1_score': MeanState(total=3.0, count=6)
              0.5,  # 'accuracy': MeanState(total=3, count=6)
              0.5,  # 'binary_accuracy': MeanState(total=3, count=6)
              0.5,  # 'sensitivity': MeanState(total=3.0, count=6)
              0.5,  # 'tpr': MeanState(total=3.0, count=6)
              0.5,  # 'specificity': MeanState(total=3.0, count=6)
              0.5,  # 'tnr': MeanState(total=3.0, count=6)
              0.5,  # 'fall_out': MeanState(total=3.0, count=6)
              0.5,  # 'fpr': MeanState(total=3.0, count=6)
              0.5,  # 'miss_rate': MeanState(total=3.0, count=6)
              0.5,  # 'fnr': MeanState(total=3.0, count=6)
              0.5,  # 'negative_prediction_value': MeanState(total=3.0, count=6)
              0.5,  # 'nvp': MeanState(total=3.0, count=6)
              0.5,  # 'false_discovery_rate': MeanState(total=3.0, count=6)
              0.5,  # 'false_omission_rate': MeanState(total=3.0, count=6)
              0.5,  # 'threat_score': MeanState(total=3.0, count=6)
              0.0,  # 'positive_likelihood_ratio': MeanState(total=0.0, count=6)
              0.0,  # 'negative_likelihood_ratio': MeanState(total=0.0, count=6)
              0.0,  # 'diagnostic_odds_ratio': MeanState(total=0.0, count=6)
              0.5,  # 'positive_predictive_value': MeanState(total=3.0, count=6)
              0.5,  # 'intersection_over_union': MeanState(total=3.0, count=6)
              0.5,  # 'prevalence': MeanState(total=3.0, count=6)
              0.5,  # 'prevalence_threshold': MeanState(total=3.0, count=6)
              0.0,  # 'matthews_correlation_coefficient':
              #   MeanState(total=0.0, count=6)
              0.0,  # 'informedness': MeanState(total=0.0, count=6)
              0.0,  # 'markedness': MeanState(total=0.0, count=6)
              0.5,  # 'balanced_accuracy': MeanState(total=3.0, count=6)
          ),
      ),
      dict(
          testcase_name="multiclass_samples",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          # precision = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8 = 0.625
          # recall = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8 = 0.625
          # f1_score = 2 * precision * recall / (precision + recall) = 0.625
          # accuracy = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8 = 0.625
          expected=(
              0.625,  # 'precision': MeanState(total=5.0, count=8)
              0.625,  # 'ppv': MeanState(total=5.0, count=8)
              0.625,  # 'recall': MeanState(total=5.0, count=8)
              0.625,  # 'f1_score': MeanState(total=5.0, count=8)
              0.625,  # 'accuracy': MeanState(total=5, count=8)
              0.75,  # 'binary_accuracy': MeanState(total=6, count=8)
              0.625,  # 'sensitivity': MeanState(total=5.0, count=8)
              0.625,  #  'tpr': MeanState(total=5.0, count=8)
              0.8125,  # 'specificity': MeanState(total=6.5, count=8)
              0.8125,  #  'tnr': MeanState(total=6.5, count=8)
              0.1875,  #  'fall_out': MeanState(total=1.5, count=8)
              0.1875,  #  'fpr': MeanState(total=1.5, count=8)
              0.375,  # 'miss_rate': MeanState(total=3.0, count=8)
              0.375,  #  'fnr': MeanState(total=3.0, count=8)
              0.8125,
              # 'negative_prediction_value': MeanState(total=6.5, count=8)
              0.8125,  # 'nvp': MeanState(total=6.5, count=8)
              0.375,  # 'false_discovery_rate': MeanState(total=3.0, count=8)
              0.1875,  # 'false_omission_rate': MeanState(total=1.5, count=8)
              0.625,  # 'threat_score': MeanState(total=5.0, count=8)
              0.0,  # 'positive_likelihood_ratio': MeanState(total=0.0, count=8)
              0.75,
              # 'negative_likelihood_ratio': MeanState(total=6.0, count=8)
              0.0,  # 'diagnostic_odds_ratio': MeanState(total=0.0, count=8)
              0.625,
              # 'positive_predictive_value': MeanState(total=5.0, count=8)
              0.625,  # 'intersection_over_union': MeanState(total=5.0, count=8)
              1 / 3,  # 'prevalence': MeanState(total=8/3, count=8)
              3 / 8,  # 'prevalence_threshold': MeanState(total=3.0, count=8)
              3.5 / 8,  # 'matthews_correlation_coefficient':
              #   MeanState(total=3.5, count=8)
              3.5 / 8,  # 'informedness': MeanState(total=3.5, count=8)
              3.5 / 8,  # 'markedness': MeanState(total=3.5, count=8)
              5.75 / 8,  # 'balanced_accuracy': MeanState(total=5.75, count=8)
          ),
      ),
      dict(
          testcase_name="multiclass_multioutput",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          # A vocab is required to align class index for macro.
          # vocab={"y": 0, "n": 1, "u": 2},
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          # precision = mean([1, 0.5, 0, 1, 1, 1, 0, 1]) = 5.5/8 = 0.6875
          # recall = mean([1, 1, 0, 1, 0.5, 1, 0, 1]) = 5.5 / 8 = 0.6875
          # f1_score = mean(preceision * recall / (precision + recall) )= 0.6667
          # accuracy = mean([1, 1, 0, 1, 1, 1, 0, 1]) = 6 / 8 = 0.75
          expected=(
              0.6875,  # 'precision': MeanState(total=5.5, count=8)
              0.6875,  # , 'ppv': MeanState(total=5.5, count=8)
              0.6875,  # 'recall': MeanState(total=5.5, count=8)
              0.6666666666666666,
              # 'f1_score': MeanState(total=5.333333333333333, count=8)
              0.75,  # 'accuracy': MeanState(total=6, count=8)
              0.75,  # 'binary_accuracy': MeanState(total=6, count=8)
              0.6875,  # 'sensitivity': MeanState(total=5.5, count=8)
              0.6875,  # 'tpr': MeanState(total=5.5, count=8)
              0.8125,  # 'specificity': MeanState(total=6.5, count=8)
              0.8125,  # 'tnr': MeanState(total=6.5, count=8)
              0.1875,  # 'fall_out': MeanState(total=1.5, count=8)
              0.1875,  # 'fpr': MeanState(total=1.5, count=8)
              0.3125,  # 'miss_rate': MeanState(total=2.5, count=8)
              0.3125,  # 'fnr': MeanState(total=2.5, count=8)
              0.8125,
              # 'negative_prediction_value': MeanState(total=6.5, count=8)
              0.8125,  # 'nvp': MeanState(total=6.5, count=8)
              0.3125,  # 'false_discovery_rate': MeanState(total=2.5, count=8)
              0.1875,  # 'false_omission_rate': MeanState(total=1.5, count=8)
              0.625,  # 'threat_score': MeanState(total=5.0, count=8)
              0.25,
              # 'positive_likelihood_ratio': MeanState(total=2.0, count=8)
              0.5625,
              # 'negative_likelihood_ratio': MeanState(total=4.5, count=8)
              0.0,  # 'diagnostic_odds_ratio': MeanState(total=0.0, count=8)
              0.6875,
              # 'positive_predictive_value': MeanState(total=5.5, count=8)
              0.625,  # 'intersection_over_union': MeanState(total=5.0, count=8)
              3 / 8,  # 'prevalence': MeanState(total=3, count=8)
              # prevalence_threshold =
              #               mean([0, (sqrt(0.5) - 0.5)/0.5, 1, 0, 0, 0, 1, 0])
              (1 / utils.pos_sqrt(0.5) + 1) / 8,  # 'prevalence_threshold':
              #                        MeanState(total=1.5 + sqrt(0.5), count=8)
              0.5,  # 'matthews_correlation_coefficient':
              #         MeanState(total=4, count=8)
              0.5,  # 'informedness': MeanState(total=4, count=8)
              0.5,  # 'markedness': MeanState(total=4, count=8)
              0.75,  # 'balanced_accuracy': MeanState(total=6, count=8)
          ),
      ),
  ])
  def test_samplewise_confusion_matrix_metric(
      self,
      y_true,
      y_pred,
      metrics,
      input_type,
      expected,
      vocab=None,
  ):
    """Tests the Samplewise Confusion Matrix for derived metrics.

    Args:
      y_true: True value array of the labels
      y_pred: Predicted value array of the labels
      metrics: Derived metric list to be calculated
      input_type: Type of label inputs Refer
        ml_metrics/_src/aggregates/types.py for values
      expected: Expected values of the derived metrics for this input
      vocab: Vocabulary of the labels
    """
    confusion_matrix = classification.SamplewiseConfusionMatrixAggFn(
        metrics=metrics,
        input_type=input_type,
        vocab=vocab,
    )
    np.testing.assert_allclose(expected, confusion_matrix(y_true, y_pred))

  def test_confusion_matrix_merge(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.ConfusionMatrixAggFn()
    baseline = classification._ConfusionMatrix(tp=1, tn=1, fp=1, fn=1)
    acc = confusion_matrix.update_state(baseline, y_true, y_pred)
    acc_other = confusion_matrix.update_state(None, y_true, y_pred)
    actual = confusion_matrix.merge_states([acc, acc_other])
    expected = classification._ConfusionMatrix(tp=5, tn=5, fp=3, fn=5)
    self.assertEqual(expected, actual)

  def test_samplewise_confusion_matrix_merge(self):
    y_pred = ["dog", "cat", "cat"]
    y_true = ["dog", "cat", "bird"]
    confusion_matrix = classification.SamplewiseConfusionMatrixAggFn(
        metrics=(ConfusionMatrixMetric.PRECISION,),
        input_type=InputType.MULTICLASS,
    )
    acc = confusion_matrix.update_state(
        confusion_matrix.create_state(), y_true, y_pred
    )
    actual = confusion_matrix.update_state(acc, y_true, y_pred)
    expected = {"precision": utils.MeanState(4, 6)}
    self.assertDictEqual(expected, actual)

  def test_confusion_matrix_add(self):
    a = classification._ConfusionMatrix(tp=4, tn=4, fp=2, fn=4)
    b = classification._ConfusionMatrix(tp=1, tn=6, fp=9, fn=2)
    expected = classification._ConfusionMatrix(tp=5, tn=10, fp=11, fn=6)
    self.assertEqual(a + b, expected)

  def test_confusion_matrix_trues_positives(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.ConfusionMatrixAggFn()
    cm = confusion_matrix(y_true, y_pred)[0]
    self.assertEqual(3, cm.p)
    self.assertEqual(4, cm.t)

  def test_confusion_matrix_invalid_input_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.ConfusionMatrixAggFn(
        input_type=InputType.CONTINUOUS
    )
    with self.assertRaisesRegex(NotImplementedError, "is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_samplewise_confusion_matrix_invalid_input_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.SamplewiseConfusionMatrixAggFn(
        metrics=(), input_type=InputType.CONTINUOUS
    )
    with self.assertRaisesRegex(NotImplementedError, "is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_invalid_average_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.ConfusionMatrixAggFn(
        average=AverageType.WEIGHTED
    )
    with self.assertRaisesRegex(NotImplementedError, " is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_binary_average_invalid_input(self):
    y_pred = ["dog", "cat", "cat", "bird", "tiger"]
    y_true = ["dog", "cat", "bird", "cat", "tiger"]
    confusion_matrix = classification.ConfusionMatrixAggFn(
        input_type=InputType.MULTICLASS_MULTIOUTPUT,
        average=AverageType.BINARY,
    )
    with self.assertRaisesRegex(ValueError, "input is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_multiclass_indicator_incorrect_input_shape(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.ConfusionMatrixAggFn(
        input_type=InputType.MULTICLASS_INDICATOR
    )
    with self.assertRaisesRegex(ValueError, "needs to be 2D array"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_multiclass_macro_vocab_reqruired(self):
    confusion_matrix = classification.ConfusionMatrixAggFn(
        average=AverageType.MACRO
    )
    acc = classification._ConfusionMatrix(tp=1, tn=1, fp=1, fn=1)
    with self.assertRaisesRegex(ValueError, "Global vocab is needed"):
      confusion_matrix.merge_states([acc])

  def test_confusion_matrix_samples_average_disallowed(self):
    with self.assertRaisesRegex(ValueError, "average is unsupported,"):
      classification.ConfusionMatrixAggFn(average=AverageType.SAMPLES)

  def test_confusion_matrix_binary_input_samples_average_disallowed(self):
    with self.assertRaisesRegex(
        ValueError, "Samples average is not available for Binary"
    ):
      classification.SamplewiseConfusionMatrixAggFn(
          metrics=(), input_type=InputType.BINARY
      )


if __name__ == "__main__":
  absltest.main()
