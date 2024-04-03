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
from absl.testing import absltest


class MathUtilsTest(absltest.TestCase):

  def test_pos_sqrt(self):
    self.assertEqual(math_utils.pos_sqrt(4), 2.0)

  def test_pos_sqrt_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError, 'Attempt to take sqrt of negative value: -1'
    ):
      math_utils.pos_sqrt(-1)


if __name__ == '__main__':
  absltest.main()
