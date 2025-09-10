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
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.utils import math_utils
from ml_metrics._src.utils import test_utils
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
    "negative_predictive_value",
    "npv",
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
    self.assertEqual(expected, confusion_matrix(y_true, y_pred))

  def test_unhashable_inputs(self):
    agg_fn = classification.ConfusionMatrixAggFn(
        input_type=InputType.MULTICLASS
    )
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0])
    y_pred = np.expand_dims(y_pred, axis=1)
    y_true = [1, 1, 0, 0, 1, 0, 1]
    with self.assertRaisesRegex(TypeError, "Unhashable elements in the input"):
      agg_fn(y_true, y_pred)

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
    self.assertEqual(expected, confusion_matrix(y_true, y_pred))

  @parameterized.named_parameters([
      # tp=2, tn=2, fp=1, fn=2, p = 3, t = 4
      dict(
          testcase_name="binary_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="binary",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          expected={
              "precision": 2 / 3,
              "ppv": 2 / 3,
              "recall": 2 / 4,
              "f1_score": 4 / 7,
              "accuracy": 1,
              "binary_accuracy": 4 / 7,
              "sensitivity": 2 / 4,
              "tpr": 2 / 4,
              "specificity": 2 / 3,
              "tnr": 2 / 3,
              "fall_out": 1 / 3,
              "fpr": 1 / 3,
              "miss_rate": 2 / 4,
              "fnr": 2 / 4,
              "negative_predictive_value": 2 / 4,
              "npv": 2 / 4,
              "false_discovery_rate": 1 / 3,
              "false_omission_rate": 2 / 4,
              "threat_score": 2 / 5,
              "positive_likelihood_ratio": 3 / 2,
              "negative_likelihood_ratio": 3 / 4,
              "diagnostic_odds_ratio": 4 / 2,
              "positive_predictive_value": 2 / 3,
              "intersection_over_union": 2 / 5,
              "prevalence": 4 / 7,
              "prevalence_threshold": math_utils.pos_sqrt(6) - 2,
              "matthews_correlation_coefficient": 1 / 6,
              "informedness": 1 / 6,
              "markedness": 1 / 6,
              "balanced_accuracy": 7 / 12,
          },
      ),
      # tp=[2 2], tn=[2 2], fp=[1 2], fn=[2 1], p = [3 4], t = [4 3]
      dict(
          testcase_name="macro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="macro",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          expected={
              "precision": (2 / 3 + 2 / 4) / 2,
              "ppv": (2 / 3 + 2 / 4) / 2,
              "recall": (2 / 4 + 2 / 3) / 2,
              "f1_score": 4 / 7,
              "accuracy": 1,
              "binary_accuracy": (4 / 7 + 4 / 7) / 2,
              "sensitivity": (2 / 4 + 2 / 3) / 2,
              "tpr": (2 / 4 + 2 / 3) / 2,
              "specificity": (2 / 3 + 2 / 4) / 2,
              "tnr": (2 / 3 + 2 / 4) / 2,
              "fall_out": (1 / 3 + 2 / 4) / 2,
              "fpr": (1 / 3 + 2 / 4) / 2,
              "miss_rate": (2 / 4 + 1 / 3) / 2,
              "fnr": (2 / 4 + 1 / 3) / 2,
              "negative_predictive_value": (2 / 4 + 2 / 3) / 2,
              "npv": (2 / 4 + 2 / 3) / 2,
              "false_discovery_rate": (1 / 3 + 2 / 4) / 2,
              "false_omission_rate": (2 / 4 + 1 / 3) / 2,
              "threat_score": (2 / 5 + 2 / 5) / 2,
              "positive_likelihood_ratio": (6 / 4 + 4 / 3) / 2,
              "negative_likelihood_ratio": (3 / 4 + 4 / 6) / 2,
              "diagnostic_odds_ratio": (6 / 3 + 6 / 3) / 2,
              "positive_predictive_value": (2 / 3 + 2 / 4) / 2,
              "intersection_over_union": (2 / 5 + 2 / 5) / 2,
              "prevalence": (4 / 7 + 3 / 7) / 2,
              "prevalence_threshold": (
                  (math_utils.pos_sqrt(6) - 2)
                  + (2 * math_utils.pos_sqrt(3) - 3)
              ) / 2,
              "matthews_correlation_coefficient": (1 / 6 + 1 / 6) / 2,
              "informedness": (1 / 6 + 1 / 6) / 2,
              "markedness": (1 / 6 + 1 / 6) / 2,
              "balanced_accuracy": (7 / 12 + 7 / 12) / 2,
          },
      ),
      # tp=2, tn=2, fp=1, fn=2, p = 3, t = 4
      dict(
          testcase_name="multiclass_indicator_binary_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="binary",
          metrics=_ImplementedDerivedConfusionMatrixMetrics,
          expected={
              "precision": 2 / 3,
              "ppv": 2 / 3,
              "recall": 2 / 4,
              "f1_score": 4 / 7,
              "accuracy": 1,
              "binary_accuracy": 4 / 7,
              "sensitivity": 2 / 4,
              "tpr": 2 / 4,
              "specificity": 2 / 3,
              "tnr": 2 / 3,
              "fall_out": 1 / 3,
              "fpr": 1 / 3,
              "miss_rate": 2 / 4,
              "fnr": 2 / 4,
              "negative_predictive_value": 2 / 4,
              "npv": 2 / 4,
              "false_discovery_rate": 1 / 3,
              "false_omission_rate": 2 / 4,
              "threat_score": 2 / 5,
              "positive_likelihood_ratio": 3 / 2,
              "negative_likelihood_ratio": 3 / 4,
              "diagnostic_odds_ratio": 4 / 2,
              "positive_predictive_value": 2 / 3,
              "intersection_over_union": 2 / 5,
              "prevalence": 4 / 7,
              "prevalence_threshold": math_utils.pos_sqrt(6) - 2,
              "matthews_correlation_coefficient": 1 / 6,
              "informedness": 1 / 6,
              "markedness": 1 / 6,
              "balanced_accuracy": 7 / 12,
          },
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
          expected={
              "precision": 2 / 3,
              "ppv": 2 / 3,
              "recall": 2 / 4,
              "f1_score": 4 / 7,
              "accuracy": 1,
              "binary_accuracy": 4 / 7,
              "sensitivity": 2 / 4,
              "tpr": 2 / 4,
              "specificity": 2 / 3,
              "tnr": 2 / 3,
              "fall_out": 1 / 3,
              "fpr": 1 / 3,
              "miss_rate": 2 / 4,
              "fnr": 2 / 4,
              "negative_predictive_value": 2 / 4,
              "npv": 2 / 4,
              "false_discovery_rate": 1 / 3,
              "false_omission_rate": 2 / 4,
              "threat_score": 2 / 5,
              "positive_likelihood_ratio": 3 / 2,
              "negative_likelihood_ratio": 3 / 4,
              "diagnostic_odds_ratio": 4 / 2,
              "positive_predictive_value": 2 / 3,
              "intersection_over_union": 2 / 5,
              "prevalence": 4 / 7,
              "prevalence_threshold": math_utils.pos_sqrt(6) - 2,
              "matthews_correlation_coefficient": 1 / 6,
              "informedness": 1 / 6,
              "markedness": 1 / 6,
              "balanced_accuracy": 7 / 12,
          },
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
    test_utils.assert_nested_container_equal(
        self, expected, confusion_matrix(y_true, y_pred)
    )

  def test_confusion_matrix_metric_invalid_average_type(self):
    with self.assertRaisesRegex(
        NotImplementedError, '"weighted" average is not supported'
    ):
      _ = classification.ConfusionMatrixAggFn(
          metrics="precision",
          average="weighted",
      )

  def test_topk_confusion_matrix_invalidinput_type(self):
    with self.assertRaisesRegex(ValueError, '"binary" input is not supported'):
      classification.TopKConfusionMatrixAggFn(input_type="binary")

  def test_confusion_matrix_metric_invalid_metric(self):
    confusion_matrix = classification.ConfusionMatrixAggFn(
        metrics="mean_average_precision"
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
          expected={
              "precision": 0.5,  # MeanState(total=3.0, count=6)
              "ppv": 0.5,  # MeanState(total=3.0, count=6)
              "recall": 0.5,  # MeanState(total=3.0, count=6)
              "f1_score": 0.5,  # MeanState(total=3.0, count=6)
              "accuracy": 0.5,  # MeanState(total=3, count=6)
              "binary_accuracy": 0.5,  # MeanState(total=3, count=6)
              "sensitivity": 0.5,  # MeanState(total=3.0, count=6)
              "tpr": 0.5,  # MeanState(total=3.0, count=6)
              "specificity": 0.5,  # MeanState(total=3.0, count=6)
              "tnr": 0.5,  # MeanState(total=3.0, count=6)
              "fall_out": 0.5,  # MeanState(total=3.0, count=6)
              "fpr": 0.5,  # MeanState(total=3.0, count=6)
              "miss_rate": 0.5,  # MeanState(total=3.0, count=6)
              "fnr": 0.5,  # MeanState(total=3.0, count=6)
              "negative_predictive_value": 0.5,  # MeanState(total=3.0, count=6)
              "npv": 0.5,  # MeanState(total=3.0, count=6)
              "false_discovery_rate": 0.5,  # MeanState(total=3.0, count=6)
              "false_omission_rate": 0.5,  # MeanState(total=3.0, count=6)
              "threat_score": 0.5,  # MeanState(total=3.0, count=6)
              "positive_likelihood_ratio": 0.0,  # MeanState(total=0.0, count=6)
              "negative_likelihood_ratio": 0.0,  # MeanState(total=0.0, count=6)
              "diagnostic_odds_ratio": 0.0,  # MeanState(total=0.0, count=6)
              "positive_predictive_value": 0.5,  # MeanState(total=3.0, count=6)
              "intersection_over_union": 0.5,  # MeanState(total=3.0, count=6)
              "prevalence": 0.5,  # MeanState(total=3.0, count=6)
              "prevalence_threshold": 0.5,  # MeanState(total=3.0, count=6)
              # MeanState(total=0.0, count=6)
              "matthews_correlation_coefficient": 0.0,
              "informedness": 0.0,  # MeanState(total=0.0, count=6)
              "markedness": 0.0,  # MeanState(total=0.0, count=6)
              "balanced_accuracy": 0.5,  # MeanState(total=3.0, count=6)
          },
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
          expected={
              "precision": 0.625,  # MeanState(total=5.0, count=8)
              "ppv": 0.625,  # MeanState(total=5.0, count=8)
              "recall": 0.625,  # MeanState(total=5.0, count=8)
              "f1_score": 0.625,  # MeanState(total=5.0, count=8)
              "accuracy": 0.625,  # MeanState(total=5, count=8)
              "binary_accuracy": 0.75,  # MeanState(total=6, count=8)
              "sensitivity": 0.625,  # MeanState(total=5.0, count=8)
              "tpr": 0.625,  # MeanState(total=5.0, count=8)
              "specificity": 0.8125,  # MeanState(total=6.5, count=8)
              "tnr": 0.8125,  # MeanState(total=6.5, count=8)
              "fall_out": 0.1875,  # MeanState(total=1.5, count=8)
              "fpr": 0.1875,  # MeanState(total=6.5, count=8)
              "miss_rate": 0.375,  # MeanState(total=3.0, count=8)
              "fnr": 0.375,  # MeanState(total=3.0, count=8)
              # MeanState(total=6.5, count=8)
              "negative_predictive_value": 0.8125,
              "npv": 0.8125,  # MeanState(total=6.5, count=8)
              "false_discovery_rate": 0.375,  # MeanState(total=3.0, count=8)
              "false_omission_rate": 0.1875,  # MeanState(total=1.5, count=8)
              "threat_score": 0.625,  # MeanState(total=5.0, count=8)
              "positive_likelihood_ratio": 0.0,  # MeanState(total=0.0, count=8)
              # MeanState(total=6.0, count=8)
              "negative_likelihood_ratio": 0.75,
              "diagnostic_odds_ratio": 0.0,  # MeanState(total=0.0, count=8)
              # MeanState(total=5.0, count=8)
              "positive_predictive_value": 0.625,
              "intersection_over_union": 0.625,  # MeanState(total=5.0, count=8)
              "prevalence": 1 / 3,  # MeanState(total=8/3, count=8)
              "prevalence_threshold": 3 / 8,  # MeanState(total=3.0, count=8)
              # MeanState(total=3.5, count=8)
              "matthews_correlation_coefficient": 3.5 / 8,
              "informedness": 3.5 / 8,  # MeanState(total=3.5, count=8)
              "markedness": 3.5 / 8,  # MeanState(total=3.5, count=8)
              "balanced_accuracy": 5.75 / 8,  # MeanState(total=5.75, count=8)
          },
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
          expected={
              "precision": 0.6875,  # MeanState(total=5.5, count=8)
              "ppv": 0.6875,  # MeanState(total=5.5, count=8)
              "recall": 0.6875,  # MeanState(total=5.5, count=8)
              "f1_score": 2 / 3,  # MeanState(total=5.33333333333333, count=8)
              "accuracy": 0.75,  # MeanState(total=6, count=8)
              "binary_accuracy": 0.75,  # MeanState(total=6, count=8)
              "sensitivity": 0.6875,  # MeanState(total=5.5, count=8)
              "tpr": 0.6875,  # MeanState(total=5.5, count=8)
              "specificity": 0.8125,  # MeanState(total=6.5, count=8)
              "tnr": 0.8125,  # MeanState(total=6.5, count=8)
              "fall_out": 0.1875,  # MeanState(total=1.5, count=8)
              "fpr": 0.1875,  # MeanState(total=1.5, count=8)
              "miss_rate": 0.3125,  # MeanState(total=2.5, count=8)
              "fnr": 0.3125,  # MeanState(total=2.5, count=8)
              # MeanState(total=6.5, count=8)
              "negative_predictive_value": 0.8125,
              "npv": 0.8125,  # MeanState(total=6.5, count=8)
              "false_discovery_rate": 0.3125,  # MeanState(total=2.5, count=8)
              "false_omission_rate": 0.1875,  # MeanState(total=1.5, count=8)
              "threat_score": 0.625,  # MeanState(total=5.0, count=8)
              "positive_likelihood_ratio": 0.25,  # MeanState(total=2, count=8)
              # MeanState(total=4.5, count=8)
              "negative_likelihood_ratio": 0.5625,
              "diagnostic_odds_ratio": 0.0,  # MeanState(total=0.0, count=8)
              # MeanState(total=5.5, count=8)
              "positive_predictive_value": 0.6875,
              "intersection_over_union": 0.625,  # MeanState(total=5.0, count=8)
              "prevalence": 3 / 8,  # MeanState(total=3, count=8)
              # MeanState(total=1.5 + sqrt(0.5), count=8)
              "prevalence_threshold": (1 / math_utils.pos_sqrt(0.5) + 1) / 8,
              # MeanState(total=4, count=8)
              "matthews_correlation_coefficient": 0.5,
              "informedness": 0.5,  # MeanState(total=4, count=8)
              "markedness": 0.5,  # MeanState(total=4, count=8)
              "balanced_accuracy": 0.75,  # MeanState(total=6, count=8)
          },
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
    confusion_matrix = classification.SamplewiseClassification(
        metrics=metrics,
        input_type=input_type,
        vocab=vocab,
    ).as_agg_fn()
    test_utils.assert_nested_container_equal(
        self,
        expected,
        confusion_matrix(y_true, y_pred),
    )

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
    confusion_matrix = classification.SamplewiseClassification(
        metrics=(ConfusionMatrixMetric.PRECISION,),
        input_type=InputType.MULTICLASS,
    ).as_agg_fn()
    state1 = confusion_matrix.update_state(
        confusion_matrix.create_state(), y_true, y_pred
    )
    state2 = confusion_matrix.update_state(
        confusion_matrix.create_state(), y_true, y_pred
    )
    actual = confusion_matrix.merge_states([state1, state2])
    expected = {"precision": utils.MeanState(4, 6)}
    assert isinstance(actual, classification.SamplewiseClassification)
    self.assertDictEqual(expected, actual.state)

  def test_confusion_matrix_add(self):
    a = classification._ConfusionMatrix(tp=4, tn=4, fp=2, fn=4)
    b = classification._ConfusionMatrix(tp=1, tn=6, fp=9, fn=2)
    expected = classification._ConfusionMatrix(tp=5, tn=10, fp=11, fn=6)
    self.assertEqual(a + b, expected)

  def test_confusion_matrix_trues_positives(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.ConfusionMatrixAggFn()
    cm = confusion_matrix(y_true, y_pred)
    self.assertEqual(3, cm.p)
    self.assertEqual(4, cm.t)

  def test_confusion_matrix_invalidinput_type(self):
    with self.assertRaisesRegex(NotImplementedError, "is not supported"):
      _ = classification.ConfusionMatrixAggFn(input_type=InputType.CONTINUOUS)

  def test_samplewise_confusion_matrix_invalidinput_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = classification.SamplewiseClassification(
        metrics=(), input_type=InputType.CONTINUOUS
    ).as_agg_fn()
    with self.assertRaisesRegex(NotImplementedError, "is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_invalid_average_type(self):
    with self.assertRaisesRegex(NotImplementedError, " is not supported"):
      _ = classification.ConfusionMatrixAggFn(average=AverageType.WEIGHTED)

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

  def test_fn_config_to_lazy_fn_by_module(self):
    actual = lazy_fns.maybe_make(
        lazy_fns.FnConfig(
            fn="SamplewiseClassification",
            module="ml_metrics._src.aggregates.classification",
            kwargs=dict(
                metrics=("recall", "precision"),
                input_type="multiclass-multioutput",
            ),
        ).make_lazy_fn()
    )
    self.assertEqual(
        classification.SamplewiseClassification(  # pytype: disable=wrong-arg-types
            metrics=("recall", "precision"),
            input_type="multiclass-multioutput",
        ),
        actual,
    )


if __name__ == "__main__":
  absltest.main()
