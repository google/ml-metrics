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
"""Tests for Math Utils."""

from ml_metrics._src.utils import math_utils
import numpy as np
from numpy import testing

from absl.testing import absltest


class MathUtilsTest(absltest.TestCase):

  def test_sigmoid(self):
    all_x = (-100, -1, -0.1, 0, 0.1, 1, 100)
    for x in all_x:
      self.assertEqual(math_utils.sigmoid(x), 1 / (1 + np.exp(-x)))

  def test_sigmoid_np_array(self):
    x = np.array((-100, -1, -0.1, 0, 0.1, 1, 100))
    testing.assert_array_equal(math_utils.sigmoid(x), 1 / (1 + np.exp(-x)))


if __name__ == '__main__':
  absltest.main()
