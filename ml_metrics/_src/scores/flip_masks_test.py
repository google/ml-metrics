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

from ml_metrics._src.scores import flip_masks
import numpy as np
from numpy import testing

from absl.testing import absltest
from absl.testing import parameterized


class FlipCountsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='binary',
          flip_mask_fn=flip_masks.binary_flip_mask,
          expected_mask=(0, 1, 1, 0),
      ),
      dict(
          testcase_name='neg_to_pos',
          flip_mask_fn=flip_masks.neg_to_pos_flip_mask,
          expected_mask=(0, 1, 0, 0),
      ),
      dict(
          testcase_name='pos_to_neg',
          flip_mask_fn=flip_masks.pos_to_neg_flip_mask,
          expected_mask=(0, 0, 1, 0),
      ),
  )
  def test_flip_counts(self, flip_mask_fn, expected_mask):
    base_predictions = np.array((0.1, 0.1, 0.9, 0.9))
    model_predictions = np.array((0.2, 0.9, 0.1, 0.8))

    for base_prediction, model_prediction, expected_flip_count in zip(
        base_predictions, model_predictions, expected_mask
    ):
      self.assertEqual(
          flip_mask_fn(base_prediction, model_prediction),
          expected_flip_count,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='binary',
          flip_mask_fn=flip_masks.binary_flip_mask,
          expected_mask=np.array((0, 1, 1, 0)),
      ),
      dict(
          testcase_name='neg_to_pos',
          flip_mask_fn=flip_masks.neg_to_pos_flip_mask,
          expected_mask=np.array((0, 1, 0, 0)),
      ),
      dict(
          testcase_name='pos_to_neg',
          flip_mask_fn=flip_masks.pos_to_neg_flip_mask,
          expected_mask=np.array((0, 0, 1, 0)),
      ),
  )
  def test_flip_counts_batched(self, flip_mask_fn, expected_mask):
    base_predictions = np.array((0.1, 0.1, 0.9, 0.9))
    model_predictions = np.array((0.2, 0.9, 0.1, 0.8))

    testing.assert_array_equal(
        flip_mask_fn(base_predictions, model_predictions), expected_mask
    )


if __name__ == '__main__':
  absltest.main()
