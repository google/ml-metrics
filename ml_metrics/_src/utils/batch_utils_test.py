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

import itertools as it
from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.utils import batch_utils
import numpy as np


def mock_range(n, batch_size, batch_fn=lambda x: x):
  """Generates (tuple of) n columns of fake data."""
  for _ in range(1000):
    yield tuple(batch_fn(np.ones(batch_size) * j) for j in range(n))
  raise ValueError(
      'Reached the end of the range, might indicate iterator is first exhausted'
      ' before running.'
  )


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='no_op',
          expected=[
              (np.zeros(2), np.ones(2)),
              (np.zeros(2), np.ones(2)),
              (np.zeros(2), np.ones(2)),
              (np.zeros(2), np.ones(2)),
              (np.zeros(2), np.ones(2)),
          ],
      ),
      dict(
          testcase_name='to_larger_batch',
          batch_size=3,
          expected=[
              (np.zeros(3), np.ones(3)),
              (np.zeros(3), np.ones(3)),
              (np.zeros(3), np.ones(3)),
              (np.zeros(1), np.ones(1)),
          ],
      ),
      dict(
          testcase_name='batch_size_is_same',
          batch_size=2,
          expected=[
              (np.zeros(2), np.ones(2)),
          ]
          * 5,
      ),
      dict(
          testcase_name='to_smaller_batch',
          batch_size=1,
          expected=[
              (np.zeros(1), np.ones(1)),
          ]
          * 10,
      ),
      dict(
          testcase_name='with_list',
          batch_size=4,
          batch_fn=list,
          expected=[
              ([0, 0, 0, 0], [1, 1, 1, 1]),
              ([0, 0, 0, 0], [1, 1, 1, 1]),
              ([0, 0], [1, 1]),
          ],
      ),
      dict(
          testcase_name='with_tuple',
          batch_size=4,
          batch_fn=tuple,
          expected=[
              ((0, 0, 0, 0), (1, 1, 1, 1)),
              ((0, 0, 0, 0), (1, 1, 1, 1)),
              ((0, 0), (1, 1)),
          ],
      ),
  ])
  def test_rebatched(self, expected, batch_size=0, batch_fn=lambda x: x):
    inputs = it.islice(mock_range(2, batch_size=2, batch_fn=batch_fn), 5)
    actual = batch_utils.rebatched(inputs, batch_size=batch_size, num_columns=2)
    for a, b in zip(expected, actual, strict=True):
      np.testing.assert_array_almost_equal(a, b)

  def test_batch_non_sequence_type(self):
    inputs = [(1, 2), (3, 4)]
    with self.assertRaisesRegex(TypeError, 'Non sequence type'):
      next(batch_utils.rebatched(iter(inputs), batch_size=4, num_columns=2))

  def test_batch_unsupported_type(self):
    inputs = [('aaa', 'bbb'), ('aaa', 'bbb')]
    with self.assertRaisesRegex(TypeError, 'Unsupported container type'):
      next(batch_utils.rebatched(iter(inputs), batch_size=4, num_columns=2))

  def test_recitable_iterator_normal(self):
    inputs = range(3)
    it_inputs = batch_utils.RecitableIterator(inputs)
    it_outputs = map(lambda x: x + 1, it_inputs)
    actual = list(zip(it_outputs, it_inputs.recite_iterator(), strict=True))
    self.assertEqual([(1, 0), (2, 1), (3, 2)], actual)

  @parameterized.named_parameters([
      dict(
          testcase_name='to_larger_batch',
          input_batch_size=2,
          fn_batch_size=5,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='to_smaller_batch',
          input_batch_size=5,
          fn_batch_size=2,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='to_same_batch_size',
          input_batch_size=3,
          fn_batch_size=3,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='to_one_element_batch',
          input_batch_size=2,
          fn_batch_size=1,
          num_columns=2,
          num_batches=30,
      ),
      dict(
          testcase_name='from_one_element_batch',
          input_batch_size=1,
          fn_batch_size=5,
          num_columns=2,
          num_batches=30,
      ),
  ])
  def test_recitable_iterator_with_rebatch(
      self,
      input_batch_size=2,
      fn_batch_size=3,
      num_columns=2,
      num_batches=5,
  ):

    def foo(columns):
      assert len(columns[0]) <= fn_batch_size, f'got {columns=}.'
      return tuple(np.array(column) + 1 for column in columns)

    inputs = it.islice(
        mock_range(num_columns, batch_size=input_batch_size), num_batches
    )
    # Setting a max buffer size to make sure the buffer is flushed while
    # iterating.
    it_inputs = batch_utils.RecitableIterator(
        inputs, max_buffer_size=max(fn_batch_size, input_batch_size)
    )
    it_fn_inputs = batch_utils.rebatched(
        it_inputs, batch_size=fn_batch_size, num_columns=num_columns
    )
    it_outputs = batch_utils.rebatched(
        map(foo, it_fn_inputs),
        batch_size=input_batch_size,
        num_columns=num_columns,
    )
    # Note that recital iterator has to be put after the original one so that
    # there are values to be recited.
    actual = zip(it_outputs, it_inputs.recite_iterator(), strict=True)
    outputs, original = zip(*actual)

    expected_orignal = list(
        it.islice(
            mock_range(num_columns, batch_size=input_batch_size), num_batches
        )
    )
    expected_outputs = [np.array(x) + 1 for x in expected_orignal]
    np.testing.assert_array_equal(expected_orignal, original)
    np.testing.assert_array_equal(expected_outputs, outputs)


if __name__ == '__main__':
  absltest.main()
