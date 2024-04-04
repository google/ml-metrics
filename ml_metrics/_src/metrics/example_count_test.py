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
"""Tests for Example Count Metric."""

from absl.testing import absltest
from ml_metrics._src.metrics import example_count

_EXAMPLES = ('example_1', 'example_2', 'example_3')


class ExampleCountTest(absltest.TestCase):

  def test_example_count_unweighted(self):
    self.assertEqual(example_count.example_count(_EXAMPLES), 3)

  def test_example_count_weighted(self):
    example_weights = (0.1, 0.3, 0.6)
    weighted_example_count = example_count.example_count(
        _EXAMPLES, example_weights
    )

    self.assertEqual(weighted_example_count, 0.1 + 0.3 + 0.6)

  def test_example_count_with_incorrect_weights(self):
    example_weights = (0.5, 0.5)

    # 3 examples and 2 example weights.
    with self.assertRaisesRegex(
        ValueError, 'examples and example_weights must have the same length'
    ):
      example_count.example_count(_EXAMPLES, example_weights)

  def test_example_count_doc(self):
    self.assertIn(
        example_count._METRIC_PYDOC_POSTFIX, example_count.example_count.__doc__
    )


if __name__ == '__main__':
  absltest.main()
