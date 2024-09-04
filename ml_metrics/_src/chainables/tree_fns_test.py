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
"""Test for tree library."""
from collections.abc import Iterable
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle as pickle
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns
import numpy as np

Key = tree.Key


class TestAverageFn:

  def __init__(self, batch_output=True, return_tuple=False):
    self.batch_output = batch_output
    self.return_tuple = return_tuple

  def create_state(self):
    return [0, 0]

  def update_state(self, state, inputs, default_value=0):
    return state[0] + sum(inputs) + default_value, state[1] + len(inputs)

  def merge_states(self, states):
    raise NotImplementedError()

  def get_result(self, state):
    result = state[0] / state[1]
    result = [result] if self.batch_output else result
    return (result, 0) if self.return_tuple else result


class TreeFnTest(parameterized.TestCase):

  def assert_nested_sequence_equal(self, a, b):
    if isinstance(a, dict) and isinstance(b, dict):
      for (k_a, v_a), (k_b, v_b) in zip(
          sorted(a.items()), sorted(b.items()), strict=True
      ):
        self.assertEqual(k_a, k_b)
        self.assert_nested_sequence_equal(v_a, v_b)
    elif isinstance(a, str) and isinstance(b, str):
      self.assertEqual(a, b)
    elif isinstance(a, Iterable) and isinstance(b, Iterable):
      for a_elem, b_elem in zip(a, b, strict=True):
        self.assert_nested_sequence_equal(a_elem, b_elem)
    else:
      self.assertEqual(a, b)

  def test_tree_fn(self):
    data = {
        'model1': {'pred': [1, (7,), 3]},
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
        'single_key': ([2, 3, 8],),
    }
    tree_fn = tree_fns.TreeFn.new(
        fn=lambda x, y: (x + 1, y + 1),
        input_keys=[Key().a, Key().c.b.at(Key.Index(0))],
        output_keys=[Key().a, Key().b],
    )
    self.assertEqual({'a': 8, 'b': 8}, tree_fn(data))

  def test_tree_fn_pass_by_kwargs(self):
    data = {
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
    }
    tree_fn = tree_fns.TreeFn.new(
        fn=lambda x, y, b: (x + b, y + b),
        input_keys=dict(x='a', y=Key().c.b.at(Key.Index(0)), b=Key.Literal(1)),
        output_keys=['a', 'b'],
    )
    self.assertEqual({'a': 8, 'b': 8}, tree_fn(data))

  def test_tree_fn_pass_by_kwargs_wrong_kwarg(self):
    data = {
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
    }
    tree_fn = tree_fns.TreeFn.new(
        fn=lambda x, y: (x + 1, y + 1),
        input_keys=dict(x='a', b=Key().c.b.at(Key.Index(0))),
        output_keys=['a', 'b'],
    )
    with self.assertRaises(ValueError):
      tree_fn(data)

  @mock.patch.object(tree_fns.TreeFn, 'actual_fn', autospec=True)
  def test_tree_fn_pickle(self, mock_actual_fn):
    tree_fn = tree_fns.TreeFn.new(
        fn=len,
        input_keys=('a', 'b'),
        output_keys=('c', 'd'),
    )
    mock_actual_fn.assert_not_called()
    self.assertEqual(tree_fn, pickle.loads(pickle.dumps(tree_fn)))

  def test_tree_fn_default_constructor_raises(self):
    with self.assertRaisesRegex(ValueError, 'Do not use the constructor.'):
      tree_fns.TreeFn()

  def test_tree_fn_assign_no_output_keys_raises(self):
    with self.assertRaisesRegex(ValueError, 'Assign should have output_keys'):
      tree_fns.Assign.new(fn=lambda: 1, output_keys=())

  def test_tree_fn_call_failed_raises(self):
    with self.assertRaisesRegex(ValueError, 'Failed to call .+ with inputs'):
      # (0,) + 1 is illegal. Should raise error.
      tree_fns.TreeFn.new(fn=lambda x: x + 1)((0,))

  def test_tree_fn_is_lazy(self):
    tree_fn = tree_fns.TreeFn.new(
        fn=lazy_fns.trace(lambda x, y: x + y)(),
    )
    self.assertTrue(tree_fn.lazy)

  def test_tree_fn_actual_fn(self):
    class Foo:

      def __call__(self):
        return 1

    tree_fn = tree_fns.TreeFn.new(
        fn=lazy_fns.trace(Foo)(),
    )
    self.assertEqual(1, tree_fn.actual_fn())

  def test_tree_fn_all_input(self):
    data = [1, 2, 3]
    tree_fn = tree_fns.TreeFn.new(
        fn=lambda x: [e + 1 for e in x],
        input_keys=Key.SELF,
    )
    self.assertEqual([2, 3, 4], tree_fn(data))

  def test_tree_fn_no_input(self):
    data = [1, 2, 3]
    tree_fn = tree_fns.TreeFn.new(fn=lambda: 1, input_keys=())
    self.assertEqual(1, tree_fn(data))

  def test_tree_fn_no_output_keys(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.TreeFn.new(
        fn=lambda x, y: (x + 1, y + 1),
        input_keys=['a', Key.new('c', 'b', Key.Index(0))],
    )
    self.assertEqual((8, 8), tree_fn(data))

  def test_assign(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.Assign.new(
        fn=lambda x, y: (x + 1, y + 1),
        input_keys=[Key().a, Key().c.b.at(Key.Index(0))],
        output_keys=['e', 'f'],
    )
    expected = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
        'e': 8,
        'f': 8,
    }
    self.assertEqual(expected, tree_fn(data))

  def test_select(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.Select.new(
        input_keys=[Key().a, Key().c.b.at(Key.Index(0))],
        output_keys=['e', 'f'],
    )
    expected = {
        'e': 7,
        'f': 7,
    }
    self.assertEqual(expected, tree_fn(data))

  def test_tree_aggfn_iterate_not_implemented(self):
    data = [[1, 2, 3], [1, 2, 3]]
    tree_fn = tree_fns.TreeAggregateFn.new(fn=TestAverageFn())
    with self.assertRaises(NotImplementedError):
      list(tree_fn.iterate(data))

  def test_tree_aggregate_fn(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn.new(fn=TestAverageFn())
    self.assertEqual([2.0], agg_fn(data))

  def test_tree_aggregate_fn_batch_output(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn.new(fn=TestAverageFn(batch_output=False))
    self.assertEqual(2.0, agg_fn(data))

  def test_tree_aggregate_fn_tuple_output(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn.new(
        fn=TestAverageFn(batch_output=False, return_tuple=True),
        output_keys=Key.SELF,
    )
    self.assertEqual((2.0, 0), agg_fn(data))

  def test_tree_aggregate_with_lazyfn(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    agg_fn = tree_fns.TreeAggregateFn.new(
        input_keys='a', fn=lazy_fns.trace(TestAverageFn)(), output_keys='mean'
    )
    self.assertEqual({'mean': [2.0]}, agg_fn(data))

  def test_tree_aggregate_with_keyword_arg(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    bias = 1
    agg_fn = tree_fns.TreeAggregateFn.new(
        input_keys=dict(inputs='a', default_value=Key.Literal(bias)),
        fn=lazy_fns.trace(TestAverageFn)(),
        output_keys='mean',
    )
    expected_mean = [(sum(data['a']) + bias) / len(data['a'])]
    self.assertEqual({'mean': expected_mean}, agg_fn(data))

  def test_tree_aggregate_with_incorrect_keyword_arg(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    agg_fn = tree_fns.TreeAggregateFn.new(
        input_keys=dict(incorrect_arg='a'),
        fn=lazy_fns.trace(TestAverageFn)(),
        output_keys='mean',
    )
    with self.assertRaises(ValueError):
      agg_fn(data)

  @parameterized.named_parameters([
      dict(
          testcase_name='default',
          inputs=[1, 2, np.array([5, 6]), 4, 5],
          expected=[1, [5], 5],
      ),
      dict(
          testcase_name='keyed_input',
          input_keys='a',
          inputs={'a': [1, 2, np.array([5, 6]), 4, 5]},
          expected=[1, [5], 5],
      ),
      dict(
          testcase_name='indexed_inputs',
          input_keys=(Key.Index(0), Key.Index(1)),
          inputs=([1, 2, np.array([5, 6]), 4, 5], [5, 6, [7, 0], 8, 9]),
          expected=([1, [5], 5], [5, [7], 9]),
      ),
      dict(
          testcase_name='replace_false',
          replace_mask_false_with=-1,
          inputs=[1, 2, np.array([5, 6]), 4, 5],
          expected=[1, -1, [5, -1], -1, 5],
      ),
      dict(
          testcase_name='replace_false_multi_inputs',
          replace_mask_false_with=-1,
          input_keys=(Key.Index(0), Key.Index(1)),
          inputs=([1, 2, np.array([5, 6]), 4, 5], [5, 6, [7, 0], 8, 9]),
          expected=([1, -1, [5, -1], -1, 5], [5, -1, [7, -1], -1, 9]),
      ),
  ])
  def test_apply_single_mask(
      self,
      inputs,
      expected,
      input_keys=Key.SELF,
      replace_mask_false_with=tree.DEFAULT_FILTER,
  ):
    masks = [True, False, [True, False], False, True]
    tree_fn = tree_fns.TreeFn.new(
        input_keys=input_keys,
        masks=masks,
        replace_mask_false_with=replace_mask_false_with,
    )
    result = tree_fn(inputs)
    self.assert_nested_sequence_equal(result, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='dict_input',
          input_keys=('a', 'b'),
          inputs={
              'a': [1, 2, np.array([5, 6]), 4, 5],
              'b': [1, 2, np.array([7, 6]), 4, 9],
          },
          expected=([1, [5], 5], [1, [7], 9]),
      ),
      dict(
          testcase_name='index_inputs',
          input_keys=(Key.Index(0), Key.Index(1)),
          inputs=([1, 2, np.array([5, 6]), 4, 5], [5, 6, [7, 0], 8, 9]),
          expected=([1, [5], 5], [5, [7], 9]),
      ),
      dict(
          testcase_name='replace_false_multi_inputs',
          replace_mask_false_with=-1,
          input_keys=(Key.Index(0), Key.Index(1)),
          inputs=([1, 2, np.array([5, 6]), 4, 5], [5, 6, [7, 0], 8, 9]),
          expected=([1, -1, [5, -1], -1, 5], [5, -1, [7, -1], -1, 9]),
      ),
  ])
  def test_apply_multi_masks(
      self,
      inputs,
      expected,
      input_keys=Key.SELF,
      replace_mask_false_with=tree.DEFAULT_FILTER,
  ):
    masks = tuple([[True, False, [True, False], False, True]] * 2)
    tree_fn = tree_fns.TreeFn.new(
        input_keys=input_keys,
        masks=masks,
        replace_mask_false_with=replace_mask_false_with,
    )
    result = tree_fn(inputs)
    self.assert_nested_sequence_equal(result, expected)


if __name__ == '__main__':
  absltest.main()
