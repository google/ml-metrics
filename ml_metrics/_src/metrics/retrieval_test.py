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
"""Tests for retrieval."""

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.metrics import retrieval
import numpy as np

InputType = retrieval.InputType
RetrievalMetric = retrieval.RetrievalMetric


class RetrievalTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="Precision",
          metric_fn=retrieval.precision,
          # precision@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
          # precision@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5/8
          expected=[5 / 8, 5.5 / 8],
      ),
      dict(
          testcase_name="PPV",
          metric_fn=retrieval.ppv,
          # ppv@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
          # ppv@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5/8
          expected=[5 / 8, 5.5 / 8],
      ),
      dict(
          testcase_name="Recall",
          metric_fn=retrieval.recall,
          # recall@1 = mean([1, 0, 0, 1, 1/2, 1, 0, 1]) = 4.5/8
          # recall@2 = mean([1, 1, 0, 1, 1/2, 1, 0, 1]) = 5.5/8
          expected=[4.5 / 8, 5.5 / 8],
      ),
      dict(
          testcase_name="Sensitivity",
          metric_fn=retrieval.sensitivity,
          # sensitivity@1 = mean([1, 0, 0, 1, 1/2, 1, 0, 1]) = 4.5/8
          # sensitivity@2 = mean([1, 1, 0, 1, 1/2, 1, 0, 1]) = 5.5/8
          expected=[4.5 / 8, 5.5 / 8],
      ),
      dict(
          testcase_name="TPR",
          metric_fn=retrieval.tpr,
          # tpr@1 = mean([1, 0, 0, 1, 1/2, 1, 0, 1]) = 4.5/8
          # tpr@2 = mean([1, 1, 0, 1, 1/2, 1, 0, 1]) = 5.5/8
          expected=[4.5 / 8, 5.5 / 8],
      ),
      dict(
          testcase_name="POSITIVE_PREDICTIVE_VALUE",
          metric_fn=retrieval.positive_predictive_value,
          # positive_predictive_value@1 = mean([1, 0, 0, 1,
          #                                     1, 1, 0, 1]) = 5/8
          # positive_predictive_value@2 = mean([1, 1/2, 0, 1,
          #                                     1, 1, 0, 1]) = 5.5/8
          expected=[5 / 8, 5.5 / 8],
      ),
      dict(
          testcase_name="INTERSECTION_OVER_UNION",
          metric_fn=retrieval.intersection_over_union,
          # intersection_over_union@1 = mean([1, 0, 0, 1,
          #                                   1/2, 1, 0, 1]) = 4.5/8
          # intersection_over_union@2 = mean([1, 1/2, 0, 1,
          #                                   1/2, 1, 0, 1]) = 5/8
          expected=[4.5 / 8, 5 / 8],
      ),
      dict(
          testcase_name="Accuracy",
          metric_fn=retrieval.accuracy,
          # accuracy@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
          # accuracy@2 = mean([1, 1, 0, 1, 1, 1, 0, 1]) = 6/8
          expected=[5 / 8, 6 / 8],
      ),
      dict(
          testcase_name="F1Score",
          metric_fn=retrieval.f1_score,
          # f1_score@1 = mean([1, 0, 0, 1, 1/1.5, 1, 0, 1]) = (4+2/3)/8
          # f1_score@2 = mean([1, 1/1.5, 0, 1, 1/1.5, 1, 0, 1]) = (4+4/3)/8
          expected=[(4 + 2 / 3) / 8, (4 + 4 / 3) / 8],
      ),
      dict(
          testcase_name="MissRate",
          metric_fn=retrieval.miss_rate,
          # miss_rate@1 = mean([0, 1, 1, 0, 1/2, 0, 1, 0]) = 3.5/8
          # miss_rate@2 = mean([0, 0, 1, 0, 1/2, 0, 1, 0]) = 2.5/8
          expected=[3.5 / 8, 2.5 / 8],
      ),
      dict(
          testcase_name="mAP",
          metric_fn=retrieval.mean_average_precision,
          # mAP@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
          # mAP@2 = mean([1, 1/2, 0, 1, 1/2, 1, 0, 1]) = 5/8
          expected=[5 / 8, 5 / 8],
      ),
      dict(
          testcase_name="MRR",
          metric_fn=retrieval.mean_reciprocal_rank,
          # mRR@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
          # mRR@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5 / 8
          expected=[5 / 8, 5.5 / 8],
      ),
      dict(
          testcase_name="DCG",
          metric_fn=retrieval.dcg_score,
          # DCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
          # DCG@2 = mean([1, 1/log2(3), 0, 1, 1, 1, 0, 1])
          #       = (5 + 1 / log2(3)) / 8
          expected=[5 / 8, (1 / np.log2(3) + 5) / 8],
      ),
      dict(
          testcase_name="NDCG",
          metric_fn=retrieval.ndcg_score,
          # NDCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
          # NDCG@2 = mean([1, 1/log2(3), 0, 1, 1/(1+1/log2(3)), 1, 0, 1])
          #        = (4 + 1 / log2(3) + 1/(1+1/log2(3))) / 8
          expected=[5 / 8, (4 + 1 / np.log2(3) + 1 / (1 + 1 / np.log2(3))) / 8],
      ),
      dict(
          testcase_name="FMI",
          metric_fn=retrieval.fowlkes_mallows_index,
          # FMI@1 = mean(sqrt([1, 0, 0, 1, 0.5, 1, 0, 1]))
          # FMI@2 = mean(sqrt([1, 0.5, 0, 1, 0.5, 1, 0, 1]))
          expected=[
              np.sqrt([1, 0, 0, 1, 0.5, 1, 0, 1]).mean(),
              np.sqrt([1, 0.5, 0, 1, 0.5, 1, 0, 1]).mean(),
          ],
      ),
      dict(
          testcase_name="FDR",
          metric_fn=retrieval.false_discovery_rate,
          # FDR@1 = mean([0, 1, 1, 0, 0, 0, 1, 0]) = 3/8
          # FDR@2 = mean([0, 1/2, 1, 0, 0, 0, 1, 0]) = 2.5/8
          expected=[3 / 8, 2.5 / 8],
      ),
      dict(
          testcase_name="ThreatScore",
          metric_fn=retrieval.threat_score,
          # threat_score@1 = mean([1, 0, 0, 1, 0.5, 1, 0, 1]) = 4.5 / 8
          # threat_score@2 = mean([0.5, 0.5, 0, 0.5, 1/3, 0.5, 0, 0.5]) =
          #       = (2.5 + 1/3) / 8
          expected=[4.5 / 8, (2.5 + 1 / 3) / 8],
      ),
  ])
  def test_individual_metric(self, metric_fn, expected):
    k_list = [1, 2]
    y_pred = [["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]]
    y_true = [["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]]
    metric_doc_details = "\n".join(
        metric_fn.__doc__.split("\n")[1:]
    ).strip()  # ignore the description line for comparison
    self.assertEqual(
        metric_doc_details, retrieval._METRIC_PYDOC_POSTFIX.strip()
    )
    np.testing.assert_allclose(
        expected,
        metric_fn(
            y_true,
            y_pred,
            k_list=k_list,
            input_type=InputType.MULTICLASS_MULTIOUTPUT,
        ),
    )


if __name__ == "__main__":
  absltest.main()
