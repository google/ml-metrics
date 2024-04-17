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

from collections.abc import Sequence

from ml_metrics._src.utils import math_utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class MathUtilsTest(parameterized.TestCase):

  def test_pos_sqrt(self):
    self.assertEqual(math_utils.pos_sqrt(4), 2.0)

  def test_pos_sqrt_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError, 'Attempt to take sqrt of negative value: -1'
    ):
      math_utils.pos_sqrt(-1)

  @parameterized.named_parameters(
      ('both_zero', 0, 0, 0.0),
      ('zero_num', 0, 10, 0.0),
      ('zero_denom', 10, 0, 0.0),
      ('float_num', 10.5, 3, 3.5),
      ('float_denom', 14, 3.5, 4.0),
      ('array_num', [2, 4, 6, 8], 2, [1.0, 2.0, 3.0, 4.0]),
      ('array_num_denom', [2, 4, 6, 8], [4, 8, 12, 16], [0.5, 0.5, 0.5, 0.5]),
  )
  def test_safe_divide(self, a, b, expected_result):
    result = math_utils.safe_divide(a, b)
    if isinstance(result, (Sequence, np.ndarray)):
      self.assertSequenceAlmostEqual(result, expected_result)
    else:
      self.assertAlmostEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
