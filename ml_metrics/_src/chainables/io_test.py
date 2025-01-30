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
    ds = io.SequenceDataSource(list(range(3)))
    actual = [list(ds.shard(i, num_shards)) for i in range(num_shards)]
    self.assertEqual(expected, actual)

  def test_sharded_sequence_len(self):
    ds = io.SequenceDataSource(list(range(3)))
    self.assertLen(ds, 3)
    self.assertLen(ds.shard(0, 1, offset=1), 2)
    self.assertLen(ds.shard(0, 2), 2)
    self.assertLen(ds.shard(0, 2, offset=1), 1)
    self.assertLen(ds.shard(1, 2), 1)

  def test_sharded_sequence_serialization(self):
    ds = io.SequenceDataSource(range(3))
    it = ds.iterate()
    self.assertEqual(0, next(it))
    ds = ds.from_state(it.state)
    self.assertEqual([1, 2], list(it))
    self.assertEqual([1, 2], list(ds))

  def test_sharded_sequence_serialization_after_shard(self):
    ds = io.SequenceDataSource(range(4))
    it = ds.shard(1, num_shards=2).iterate()
    self.assertEqual(2, next(it))
    ds = ds.iterate().from_state(it.state)
    self.assertEqual([3], list(it))
    self.assertEqual([3], list(ds))

  def test_sharded_sequence_with_non_indexable_data(self):
    with self.assertRaisesRegex(TypeError, 'data is not indexable'):
      io.SequenceDataSource(0)  # pytype: disable=wrong-arg-types

  def test_sharded_sequence_with_invalid_num_shards_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'num_shards must be positive'):
      _ = list(io.SequenceDataSource(range(3), io.ShardConfig(num_shards=0)))

  def test_sharded_iterable(self):
    ds = io.ShardedIterable(range(3))
    self.assertEqual([0, 1, 2], list(ds))

  def test_sharded_iterable_shard(self):
    ds = io.ShardedIterable(range(3))
    num_shards = 2
    actual = [list(ds.shard(i, num_shards)) for i in range(num_shards)]
    expected = [[0, 2], [1]]
    self.assertEqual(expected, actual)

  def test_sharded_iterable_num_shards_more_than_data(self):
    ds = io.ShardedIterable(range(2))
    num_shards = 3
    actual = [list(ds.shard(i, num_shards)) for i in range(num_shards)]
    expected = [[0], [1], []]
    self.assertEqual(expected, actual)

  def test_sharded_iterable_serialization(self):
    ds = io.ShardedIterable(range(3))
    it = ds.iterate()
    self.assertEqual(0, next(it))
    ds = ds.iterate().from_state(it.state)
    self.assertEqual([1, 2], list(it))
    self.assertEqual([1, 2], list(ds))

  def test_sharded_iterable_shard_serialization(self):
    ds = io.ShardedIterable(range(6))
    it = ds.shard(1, num_shards=2).iterate()
    self.assertEqual(1, next(it))
    ds = ds.from_state(it.state)
    self.assertEqual([3, 5], list(it))
    self.assertEqual([3, 5], list(ds))

  def test_sharded_iterator_with_iterator_raises_error(self):
    with self.assertRaisesRegex(
        TypeError, 'input has to be an iterable but not an iterator'
    ):
      _ = list(io.ShardedIterable(iter(range(3))))

  def test_sharded_iterator_with_non_iteratable_raises_error(self):
    with self.assertRaisesRegex(
        TypeError, 'input has to be an iterable but not an iterator'
    ):
      _ = io.ShardedIterable(3)  # pytype: disable=wrong-arg-types

  def test_sharded_iterator_with_invalid_num_shards_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'num_shards must be positive'):
      _ = list(io.ShardedIterable(range(3), io.ShardConfig(num_shards=0)))


if __name__ == '__main__':
  absltest.main()
