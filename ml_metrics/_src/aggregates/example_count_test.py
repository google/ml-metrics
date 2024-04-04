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
from ml_metrics._src.aggregates import example_count

_EXAMPLES = ('example_1', 'example_2', 'example_3')


class ExampleCountTest(absltest.TestCase):

  def test_example_count_unweighted(self):
    self.assertEqual(example_count.ExampleCountAggFn()(_EXAMPLES), 3)

  def test_example_count_weighted(self):
    example_weights = (0.1, 0.3, 0.6)
    weighted_example_count = example_count.ExampleCountAggFn()(
        _EXAMPLES, example_weights
    )

    self.assertEqual(weighted_example_count, sum(example_weights))

  def test_example_count_with_incorrect_weights_raises_value_error(self):
    example_weights = (0.5, 0.5)
    example_count_agg_fn = example_count.ExampleCountAggFn()

    # 3 examples and 2 example weights.
    with self.assertRaisesRegex(
        ValueError, 'examples and example_weights must have the same length'
    ):
      example_count_agg_fn(_EXAMPLES, example_weights)

  def test_example_count_merge_states(self):
    example_counts = (0.2, 0.3, 0.4)
    merged_state = example_count.ExampleCountAggFn().merge_states(
        example_counts
    )

    self.assertAlmostEqual(merged_state, sum(example_counts))


if __name__ == '__main__':
  absltest.main()
