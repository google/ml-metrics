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
from ml_metrics._src.aggregates import retrieval
from ml_metrics._src.aggregates import types
from ml_metrics._src.aggregates import utils
import numpy as np


InputType = types.InputType
RetrievalMetric = retrieval.RetrievalMetric


class ClassificationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="multiclass",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type=InputType.MULTICLASS,
          metrics=[
              RetrievalMetric.FOWLKES_MALLOWS_INDEX,
              RetrievalMetric.THREAT_SCORE,
              RetrievalMetric.DCG_SCORE,
              RetrievalMetric.NDCG_SCORE,
              RetrievalMetric.MEAN_RECIPROCAL_RANK,
              RetrievalMetric.PRECISION,
              RetrievalMetric.PPV,
              RetrievalMetric.FALSE_DISCOVERY_RATE,
              RetrievalMetric.MEAN_AVERAGE_PRECISION,
              RetrievalMetric.RECALL,
              RetrievalMetric.SENSITIVITY,
              RetrievalMetric.TPR,
              RetrievalMetric.POSITIVE_PREDICTIVE_VALUE,
              RetrievalMetric.INTERSECTION_OVER_UNION,
              RetrievalMetric.MISS_RATE,
              RetrievalMetric.F1_SCORE,
              RetrievalMetric.ACCURACY,
          ],
          k_list=[1, 2],
          expected=([
              # FMI@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # FMI@2 = FMI@1 because there is only one output.
              [5 / 8, 5 / 8],
              # threat_score@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # threat_score@2 = threat_score@1 since there is only one output.
              [5 / 8, 5 / 8],
              # DCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # DCG@2 = DCG@1 since there is only one output.
              [5 / 8, 5 / 8],
              # NDCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # NDCG@2 = NDCG@1 since there is only one output.
              [5 / 8, 5 / 8],
              # mRR@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # mRR@2 = mRR@1 since there is only one output.
              [5 / 8, 5 / 8],
              # precision@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # precision@2 = precision@1 since there is only one output.
              [5 / 8, 5 / 8],
              # ppv@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # ppv@2 = ppv@1 since there is only one output.
              [5 / 8, 5 / 8],
              # FDR@1 = mean([0, 1, 1, 0, 0, 0, 1, 0]) = 3 / 8
              # FDR@2 = FDR@1 since there is only one output.
              [3 / 8, 3 / 8],
              # mAP@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # mAP@2 = mAP@1 since there is only one output.
              [5 / 8, 5 / 8],
              # recall@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # recall@2 = recall@1 since there is only one output.
              [5 / 8, 5 / 8],
              # sensitivity@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # sensitivity@2 = sensitivity@1 since there is only one output.
              [5 / 8, 5 / 8],
              # tpr@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # tpr@2 = tpr@1 since there is only one output.
              [5 / 8, 5 / 8],
              # positive_predictive_value@1 = mean([1, 0, 0, 1,
              #                                     1, 1, 0, 1]) = 5/8
              # positive_predictive_value@2 = positive_predictive_value@1
              #                               since there is only one output.
              [5 / 8, 5 / 8],
              # intersection_over_union@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # intersection_over_union@2 = intersection_over_union@1
              #                             since there is only one output.
              [5 / 8, 5 / 8],
              # miss_rate@1 = mean([0, 1, 1, 0, 0, 0, 1, 0]) = 3 / 8
              # miss_rate@2 = miss_rate@1 since there is only one output.
              [3 / 8, 3 / 8],
              # f1_score@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # f1_score@2 = f1_score@1 since there is only one output.
              [5 / 8, 5 / 8],
              # accuracy@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # accuracy@2 = accuracy@1 since there is only one output.
              [5 / 8, 5 / 8],
          ]),
      ),
      dict(
          testcase_name="multiclass_multioutput",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type=InputType.MULTICLASS_MULTIOUTPUT,
          metrics=[
              RetrievalMetric.FOWLKES_MALLOWS_INDEX,
              RetrievalMetric.THREAT_SCORE,
              RetrievalMetric.DCG_SCORE,
              RetrievalMetric.NDCG_SCORE,
              RetrievalMetric.MEAN_RECIPROCAL_RANK,
              RetrievalMetric.PRECISION,
              RetrievalMetric.PPV,
              RetrievalMetric.FALSE_DISCOVERY_RATE,
              RetrievalMetric.MEAN_AVERAGE_PRECISION,
              RetrievalMetric.RECALL,
              RetrievalMetric.SENSITIVITY,
              RetrievalMetric.TPR,
              RetrievalMetric.POSITIVE_PREDICTIVE_VALUE,
              RetrievalMetric.INTERSECTION_OVER_UNION,
              RetrievalMetric.MISS_RATE,
              RetrievalMetric.F1_SCORE,
              RetrievalMetric.ACCURACY,
          ],
          k_list=[1, 2],
          expected=([
              # FMI@1 = mean(sqrt([1, 0, 0, 1, 0.5, 1, 0, 1]))
              # FMI@2 = mean(sqrt([1, 0.5, 0, 1, 0.5, 1, 0, 1]))
              [
                  np.sqrt([1, 0, 0, 1, 0.5, 1, 0, 1]).mean(),
                  np.sqrt([1, 0.5, 0, 1, 0.5, 1, 0, 1]).mean(),
              ],
              # threat_score@1 = mean([1, 0, 0, 1, 0.5, 1, 0, 1]) = 4.5 / 8
              # threat_score@2 = mean([0.5, 0.5, 0, 0.5, 1/3, 0.5, 0, 0.5]) =
              #       = (2.5 + 1/3) / 8
              [4.5 / 8, (2.5 + 1 / 3) / 8],
              # DCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # DCG@2 = mean([1, 1/log2(3), 0, 1, 1, 1, 0, 1])
              #       = (5 + 1 / log2(3)) / 8
              [5 / 8, (1 / np.log2(3) + 5) / 8],
              # NDCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # NDCG@2 = mean([1, 1/log2(3), 0, 1, 1/(1+1/log2(3)), 1, 0, 1])
              #        = (4 + 1 / log2(3) + 1/(1+1/log2(3))) / 8
              [5 / 8, (4 + 1 / np.log2(3) + 1 / (1 + 1 / np.log2(3))) / 8],
              # mRR@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # mRR@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5 / 8
              [5 / 8, 5.5 / 8],
              # precision@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # precision@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5/8
              [5 / 8, 5.5 / 8],
              # ppv@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # ppv@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5/8
              [5 / 8, 5.5 / 8],
              # FDR@1 = mean([0, 1, 1, 0, 0, 0, 1, 0]) = 3/8
              # FDR@2 = mean([0, 1/2, 1, 0, 0, 0, 1, 0]) = 2.5/8
              [3 / 8, 2.5 / 8],
              # mAP@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # mAP@2 = mean([1, 1/2, 0, 1, 1/2, 1, 0, 1]) = 5/8
              [5 / 8, 5 / 8],
              # recall@1 = mean([1, 0, 0, 1, 1/2, 1, 0, 1]) = 4.5/8
              # recall@2 = mean([1, 1, 0, 1, 1/2, 1, 0, 1]) = 5.5/8
              [4.5 / 8, 5.5 / 8],
              # senstivity@1 = mean([1, 0, 0, 1, 1/2, 1, 0, 1]) = 4.5/8
              # senstivity@2 = mean([1, 1, 0, 1, 1/2, 1, 0, 1]) = 5.5/8
              [4.5 / 8, 5.5 / 8],
              # tpr@1 = mean([1, 0, 0, 1, 1/2, 1, 0, 1]) = 4.5/8
              # tpr@2 = mean([1, 1, 0, 1, 1/2, 1, 0, 1]) = 5.5/8
              [4.5 / 8, 5.5 / 8],
              # positive_predictive_value@1 = mean([1, 0, 0, 1,
              #                                     1, 1, 0, 1]) = 5/8
              # positive_predictive_value@2 = mean([1, 1/2, 0, 1,
              #                                     1, 1, 0, 1]) = 5.5/8
              [5 / 8, 5.5 / 8],
              # intersection_over_union@1 = mean([1, 0, 0, 1,
              #                                   1/2, 1, 0, 1]) = 4.5/8
              # intersection_over_union@2 = mean([1, 1/2, 0, 1,
              #                                   1/2, 1, 0, 1]) = 5/8
              [4.5 / 8, 5 / 8],
              # miss_rate@1 = mean([0, 1, 1, 0, 1/2, 0, 1, 0]) = 3.5/8
              # miss_rate@2 = mean([0, 0, 1, 0, 1/2, 0, 1, 0]) = 2.5/8
              [3.5 / 8, 2.5 / 8],
              # f1_score@1 = mean([1, 0, 0, 1, 1/1.5, 1, 0, 1]) = (4+2/3)/8
              # f1_score@2 = mean([1, 1/1.5, 0, 1, 1/1.5, 1, 0, 1]) = (4+4/3)/8
              [(4 + 2 / 3) / 8, (4 + 4 / 3) / 8],
              # accuracy@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # accuracy@2 = mean([1, 1, 0, 1, 1, 1, 0, 1]) = 6/8
              [5 / 8, 6 / 8],
          ]),
      ),
      dict(
          testcase_name="multiclass_multioutput_infinity_k",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type=InputType.MULTICLASS_MULTIOUTPUT,
          metrics=[RetrievalMetric.PRECISION],
          k_list=None,
          # precision = mean([1, 1/2, 0, 1, 0, 1, 0, 1]) = 5.5 / 8
          expected=([[5.5 / 8]]),
      ),
  ])
  def test_computeretrieval_metric(
      self,
      y_true,
      y_pred,
      input_type,
      metrics,
      k_list,
      expected,
  ):
    topk_retrievals = []
    # TopKRetrieval's mergeable state is MeanState, merging multiple of the same
    # state does not change the result since sum * N / cnt * N == sum / cnt.
    for _ in range(3):
      topk_retrieval = retrieval.TopKRetrievalConfig(
          metrics=metrics,
          k_list=k_list,
          input_type=input_type,
      ).make()
      topk_retrieval.add(y_true, y_pred)
      topk_retrievals.append(topk_retrieval)
    topk_retrievals[0].merge(topk_retrievals[1])
    np.testing.assert_allclose(expected, topk_retrievals[0].result())

    topk_retrieval = retrieval.TopKRetrievalConfig(
        metrics=metrics,
        k_list=k_list,
        input_type=input_type,
    ).make()
    for _ in range(3):
      topk_retrieval.add(y_true, y_pred)
    np.testing.assert_allclose(expected, topk_retrievals[0].result())

    topk_retrieval_agg_fn = retrieval.TopKRetrievalAggFn(
        metrics=metrics,
        k_list=k_list,
        input_type=input_type,
    )
    np.testing.assert_allclose(expected, topk_retrieval_agg_fn(y_true, y_pred))

  def test_retrieval_metric_add(self):
    y_pred = [["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]]
    y_true = [["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]]
    topk_retrieval = retrieval.TopKRetrievalConfig(
        metrics=[RetrievalMetric.PRECISION],
        k_list=None,
        input_type=InputType.MULTICLASS_MULTIOUTPUT,
    ).make()
    topk_retrieval.add(y_true, y_pred)
    topk_retrieval.add(y_true, y_pred)
    expected = {"precision": utils.MeanState(11, 16)}
    self.assertDictEqual(expected, topk_retrieval.state)

  def test_retrieval_metric_merge(self):
    y_pred = [["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]]
    y_true = [["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]]
    topk_retrieval1 = retrieval.TopKRetrievalConfig(
        metrics=[RetrievalMetric.PRECISION],
        k_list=None,
        input_type=InputType.MULTICLASS_MULTIOUTPUT,
    ).make()
    topk_retrieval2 = retrieval.TopKRetrievalConfig(
        metrics=[RetrievalMetric.PRECISION],
        k_list=None,
        input_type=InputType.MULTICLASS_MULTIOUTPUT,
    ).make()
    topk_retrieval1.add(y_true, y_pred)
    topk_retrieval2.add(y_true, y_pred)
    topk_retrieval1.merge(topk_retrieval2)
    expected = {"precision": utils.MeanState(11, 16)}
    self.assertDictEqual(expected, topk_retrieval1.state)


if __name__ == "__main__":
  absltest.main()
