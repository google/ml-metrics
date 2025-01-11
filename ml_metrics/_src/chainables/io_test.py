# Copyright 2025 Google LLC
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
from ml_metrics._src.chainables import io


class ShardedDataSourceTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='with_default_num_shards',
          num_shards=1,
          expected=[[0, 1, 2]],
      ),
      dict(
          testcase_name='with_two_shards',
          num_shards=2,
          expected=[[0, 1], [2]],
      ),
      dict(
          testcase_name='with_three_shards',
          num_shards=3,
          expected=[[0], [1], [2]],
      ),
      dict(
          testcase_name='with_four_shards',
          num_shards=4,
          expected=[[0], [1], [2], []],
      ),
  ])
  def test_sharded_sequence(self, num_shards, expected):
    ds = io.ShardedSequence(list(range(3)))
    actual = [list(ds.get_shard(i, num_shards)) for i in range(num_shards)]
    self.assertEqual(expected, actual)

  def test_sharded_sequence_with_non_indexable_data(self):
    with self.assertRaises(TypeError):
      io.ShardedSequence(0)  # pytype: disable=wrong-arg-types


if __name__ == '__main__':
  absltest.main()
