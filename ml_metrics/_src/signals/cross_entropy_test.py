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
"""Tests for Cross Entropy."""

import math

from ml_metrics._src.signals import cross_entropy
import numpy as np

from absl.testing import absltest


_COMPARED_DECIMAL_PLACES = 6  # Used to tune tests accuracy.


class CrossEntropyTest(absltest.TestCase):

  def test_binary_cross_entropy(self):
    y_true = np.array((0, 1, 0, 1, 0, 1, 0, 1))
    y_pred = np.array((0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9))

    # How to reproduce expected_result:
    # bce = tensorflow.keras.losses.BinaryCrossentropy()
    # expected_result = tensorflow.get_static_value(
    #     bce(y_true=y_true, y_pred=y_pred)
    # )
    expected_result = 0.9587651091286978

    self.assertAlmostEqual(
        cross_entropy.binary_cross_entropy(y_true=y_true, y_pred=y_pred),
        expected_result,
        places=_COMPARED_DECIMAL_PLACES,
    )

  def test_binary_cross_entropy_y_pred_includes_0_and_1(self):
    y_true = np.array((0, 1))
    y_pred = np.array((0, 1))

    self.assertTrue(
        math.isnan(
            cross_entropy.binary_cross_entropy(y_true=y_true, y_pred=y_pred)
        ),
    )

  def test_binary_cross_entropy_invalid_y_true_raises_error(self):
    y_true = np.array((0, 1, 0, 1, 0, 1, 0, 2))
    y_pred = np.array((0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9))

    with self.assertRaisesRegex(
        ValueError, 'y_true must contain only 0s and 1s, but recieved: '
    ):
      cross_entropy.binary_cross_entropy(y_true=y_true, y_pred=y_pred)

  def test_categorical_cross_entropy(self):
    y_true = np.array((0, 1, 0, 1, 0, 1, 0, 1))
    y_pred = np.array((0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9))

    # How to reproduce expected_result:
    # cce = tensorflow.keras.losses.CategoricalCrossentropy()
    # expected_result = tensorflow.get_static_value(
    #     cce(y_true=y_true, y_pred=y_pred)
    # )
    expected_result = 9.38023940877158

    self.assertAlmostEqual(
        cross_entropy.categorical_cross_entropy(y_true=y_true, y_pred=y_pred),
        expected_result,
        places=_COMPARED_DECIMAL_PLACES,
    )

  def test_categorical_cross_entropy_invalid_y_true_raises_error(self):
    y_true = np.array((0, 1, 0, 1, 0, 1, 0, 2))
    y_pred = np.array((0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9))

    with self.assertRaisesRegex(
        ValueError, 'y_true must contain only 0s and 1s, but recieved: '
    ):
      cross_entropy.categorical_cross_entropy(y_true=y_true, y_pred=y_pred)


if __name__ == '__main__':
  absltest.main()
