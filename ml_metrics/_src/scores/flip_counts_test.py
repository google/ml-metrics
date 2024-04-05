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
"""Tests for Flip Counts."""

from ml_metrics._src.scores import flip_counts
import numpy as np
from numpy import testing

from absl.testing import absltest
from absl.testing import parameterized


class FlipCountsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='low_base_low_model',
          base_prediction=0.1,
          model_prediction=0.2,
          symmetric_flip_count=0,
          neg_to_neg_flip_count=1,
      ),
      dict(
          testcase_name='low_base_high_model',
          base_prediction=0.1,
          model_prediction=0.9,
          symmetric_flip_count=1,
          neg_to_neg_flip_count=0,
      ),
      dict(
          testcase_name='high_base_low_model',
          base_prediction=0.9,
          model_prediction=0.1,
          symmetric_flip_count=1,
          neg_to_neg_flip_count=0,
      ),
      dict(
          testcase_name='high_base_high_model',
          base_prediction=0.9,
          model_prediction=0.8,
          symmetric_flip_count=0,
          neg_to_neg_flip_count=0,
      ),
  )
  def test_flip_counts(
      self,
      base_prediction,
      model_prediction,
      symmetric_flip_count,
      neg_to_neg_flip_count,
  ):
    self.assertEqual(
        flip_counts.flip_counts(base_prediction, model_prediction),
        symmetric_flip_count,
    )
    self.assertEqual(
        flip_counts.neg_to_neg_flip_counts(base_prediction, model_prediction),
        neg_to_neg_flip_count,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='symmetric',
          flip_counts_fn=flip_counts.flip_counts,
          expected_flip_counts=np.array((0, 1, 1, 0)),
      ),
      dict(
          testcase_name='neg_to_neg',
          flip_counts_fn=flip_counts.neg_to_neg_flip_counts,
          expected_flip_counts=np.array((1, 0, 0, 0)),
      ),
  )
  def test_flip_counts_batched(self, flip_counts_fn, expected_flip_counts):
    base_predictions = np.array((0.1, 0.1, 0.9, 0.9))
    model_predictions = np.array((0.2, 0.9, 0.1, 0.8))

    testing.assert_array_equal(
        flip_counts_fn(base_predictions, model_predictions),
        expected_flip_counts,
    )


if __name__ == '__main__':
  absltest.main()
