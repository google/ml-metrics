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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle as pickle
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns
from ml_metrics._src.utils import test_utils
import numpy as np

Key = tree.Key


class TestAverageMetric:

  def as_agg_fn(self):
    return TestAverageFn()


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


class TestAverageFnDictOutput(TestAverageFn):

  def get_result(self, state):
    return {'mean': super().get_result(state), 'state': state}


class TreeFnTest(parameterized.TestCase):

  def test_tree_fn(self):
    data = {
        'model1': {'pred': [1, (7,), 3]},
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
        'single_key': ([2, 3, 8],),
    }
    tree_fn = tree_fns.TreeFn(
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
    tree_fn = tree_fns.TreeFn(
        fn=lambda x, y, b: (x + b, y + b),
        input_keys=dict(x='a', y=Key().c.b.at(0), b=Key.Literal(1)),
        output_keys=['a', 'b'],
    )
    self.assertEqual({'a': 8, 'b': 8}, tree_fn(data))

  def test_tree_fn_pass_by_kwargs_wrong_kwarg(self):
    data = {
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
    }
    tree_fn = tree_fns.TreeFn(
        fn=lambda x, y: (x + 1, y + 1),
        input_keys=dict(x='a', b=Key().c.b.at(Key.Index(0))),
        output_keys=['a', 'b'],
    )
    with self.assertRaises(ValueError):
      tree_fn(data)

  def test_tree_fn_pass_output_keys_by_kwargs(self):
    data = {
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
    }
    tree_fn = tree_fns.TreeFn(
        fn=lambda x, y: {'o1': x + 1, 'o2': y + 1},
        input_keys=('a', Key().c.b.at(0)),
        output_keys=dict(a='o1', b='o2'),
    )
    self.assertEqual({'a': 8, 'b': 8}, tree_fn(data))

  def test_tree_fn_pass_output_keys_by_tuple_with_kwargs(self):
    data = {
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
    }
    tree_fn = tree_fns.TreeFn(
        fn=lambda x, y: ({'o1': x + 1, 'o2': y + 1}, {'o3': x + 2}, 10),
        input_keys=('a', Key().c.b.at(0)),
        output_keys=(dict(a='o1', b='o2'), dict(c='o3'), 'd'),
    )
    self.assertEqual({'a': 8, 'b': 8, 'c': 9, 'd': 10}, tree_fn(data))

  def test_tree_fn_pass_output_keys_by_kwargs_wrong_kwarg(self):
    data = {
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [([2, 3, 8],)]},
    }
    tree_fn = tree_fns.TreeFn(
        fn=lambda x, y: {'o1': x + 1, 'o2': y + 1},
        input_keys=('a', Key().c.b.at(0)),
        output_keys=dict(a='o3', b='o2'),
    )
    with self.assertRaises(KeyError):
      tree_fn(data)

  @mock.patch.object(tree_fns.TreeFn, '_actual_fn', autospec=True)
  def test_tree_fn_pickle(self, mock_actual_fn):
    tree_fn = tree_fns.TreeFn(
        fn=len,
        input_keys=('a', 'b'),
        output_keys=('c', 'd'),
    )
    mock_actual_fn.assert_not_called()
    self.assertEqual(tree_fn, pickle.loads(pickle.dumps(tree_fn)))

  def test_tree_fn_negative_batch_size_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'batch sizes have to be non-negative'
    ):
      tree_fns.TreeFn(batch_size=-1)

  def test_tree_fn_pickle_with_lazy_fn(self):
    tree_fn = tree_fns.TreeFn(fn=lazy_fns.trace(test_utils.Unpickleable)())
    self.assertEqual(3, tree_fn(2))
    # Once the function is called, it cached the result of the lazy function.
    # Thus making it different from the original, we are testing whether
    # pickling and unpickling it will restore the original state.
    tree_fn = pickle.loads(pickle.dumps(tree_fn))
    self.assertEqual(tree_fn, pickle.loads(pickle.dumps(tree_fn)))
    self.assertEqual(3, tree_fn(2))

  def test_tree_fn_ignore_error(self):
    def foo(x):
      if x == 2:
        raise ValueError('foo')
      return x

    tree_fn = tree_fns.TreeFn(
        fn=foo,
        ignore_error=True,
    )
    actual = list(tree_fn.iterate(range(5)))
    self.assertEqual([0, 1, 3, 4], actual)

  def test_tree_fn_assign_no_output_keys_raises(self):
    with self.assertRaises(AssertionError):
      tree_fns.Assign(fn=lambda: 1, output_keys=())

  def test_tree_fn_input_keys_key_error_raises(self):
    with self.assertRaisesRegex(KeyError, 'Failed to get inputs'):
      _ = tree_fns.TreeFn(fn=lambda x: x + 1, input_keys='a')({'b': 1})

  def test_tree_fn_call_failed_raises(self):
    with self.assertRaisesRegex(ValueError, 'Failed to call .+ with inputs'):
      # (0,) + 1 is illegal. Should raise error.
      tree_fns.TreeFn(fn=lambda x: x + 1)((0,))

  def test_tree_fn_is_lazy(self):
    tree_fn = tree_fns.TreeFn(
        fn=lazy_fns.trace(lambda x, y: x + y)(),
    )
    self.assertTrue(tree_fn._lazy)

  def test_tree_fn_actual_fn(self):
    class Foo:

      def __call__(self):
        return 1

    tree_fn = tree_fns.TreeFn(
        fn=lazy_fns.trace(Foo)(),
    )
    self.assertEqual(1, tree_fn._actual_fn())

  def test_tree_fn_all_input(self):
    data = [1, 2, 3]
    tree_fn = tree_fns.TreeFn(
        fn=lambda x: [e + 1 for e in x],
        input_keys=Key.SELF,
    )
    self.assertEqual([2, 3, 4], tree_fn(data))

  def test_tree_fn_no_input(self):
    data = [1, 2, 3]
    tree_fn = tree_fns.TreeFn(fn=lambda: 1, input_keys=())
    self.assertEqual(1, tree_fn(data))

  def test_tree_fn_no_output_keys(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.TreeFn(
        fn=lambda x, y: (x + 1, y + 1),
        input_keys=['a', Key().c.b.at(0)],
    )
    self.assertEqual((8, 8), tree_fn(data))

  def test_flatten_default_fn(self):
    data = [range(1), range(2), range(3)]
    tree_fn = tree_fns.Flatten()
    self.assertEqual([0, 0, 1, 0, 1, 2], list(tree_fn.iterate(data)))

  def test_flatten_fn(self):
    def foo(x):
      yield from range(x)

    data = [{'a': 1}, {'a': 2}, {'a': 3}]
    tree_fn = tree_fns.Flatten(fn=foo, input_keys='a')
    self.assertEqual([0, 0, 1, 0, 1, 2], list(tree_fn.iterate(data)))

  def test_filter_fn(self):
    data = range(6)
    tree_fn = tree_fns.Filter(fn=lambda x: x % 2 == 0)
    self.assertEqual([0, 2, 4], list(tree_fn.iterate(data)))

  def test_filter_fn_empty_output_keys(self):
    data = range(6)
    tree_fn = tree_fns.Filter(fn=lambda x: x % 2 == 0, output_keys=())
    self.assertEqual([0, 2, 4], list(tree_fn.iterate(data)))

  def test_assign(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.Assign(
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

  def test_assign_with_kw_output_keys(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.Assign(
        fn=lambda x, y: {'x': x + 1, 'y': y + 1},
        input_keys=[Key().a, Key().c.b.at(Key.Index(0))],
        output_keys=dict(g='x', h='y'),
    )
    expected = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
        'g': 8,
        'h': 8,
    }
    self.assertEqual(expected, tree_fn(data))

  def test_assign_multiple_outputs_single_key(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.Assign(
        fn=lambda x, y: (x + 1, y + 1),
        input_keys=[Key().a, Key().c.b.at(0)],
        output_keys='e',
    )
    expected = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
        'e': (8, 8)
    }
    self.assertEqual(expected, tree_fn(data))

  def test_select(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.Select(
        input_keys=[Key().a, Key().c.b.at(Key.Index(0))],
        output_keys=['e', 'f'],
    )
    expected = {
        'e': 7,
        'f': 7,
    }
    self.assertEqual(expected, tree_fn(data))

  def test_select_with_kw_output_keys(self):
    data = {
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
    }
    tree_fn = tree_fns.Select(
        input_keys='c',
        output_keys=dict(e='b', f=Key().b.at(1)),
    )
    expected = {
        'e': (7, 8),
        'f': 8,
    }
    self.assertEqual(expected, tree_fn(data))

  def test_sink_default(self):
    data = [1, 2, 3]
    sink = test_utils.TestSink()
    fn = tree_fns.Sink(fn=sink)
    self.assertIsInstance(fn._actual_fn, tree_fns._CallableSink)
    self.assertEmpty(sink.data)
    it_ = fn.iterate(data)
    self.assertFalse(sink.closed)
    self.assertEqual([1, 2, 3], list(it_))
    self.assertEqual([1, 2, 3], sink.data)
    self.assertTrue(sink.closed)

  def test_sink_lazy(self):
    data = [1, 2, 3]
    sink = lazy_fns.trace(test_utils.TestSink)(cache_result_=True)
    fn = tree_fns.Sink(fn=sink)
    self.assertEqual([1, 2, 3], list(fn.iterate(data)))
    self.assertEqual([1, 2, 3], sink.result_().data)
    self.assertTrue(sink.result_().closed)

  def test_sink_with_input_keys(self):
    data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    sink = test_utils.TestSink()
    fn = tree_fns.Sink(fn=sink, input_keys='b')
    self.assertEqual(data, list(fn.iterate(data)))
    self.assertTrue(sink.closed)
    self.assertEqual([2, 4], sink.data)

  def test_tree_aggfn_iterate_not_implemented(self):
    data = [[1, 2, 3], [1, 2, 3]]
    tree_fn = tree_fns.TreeAggregateFn(fn=TestAverageFn())
    with self.assertRaises(NotImplementedError):
      list(tree_fn.iterate(data))

  def test_tree_aggregate_fn(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn(fn=TestAverageFn())
    self.assertEqual([2.0], agg_fn(data))

  def test_tree_aggregate_fn_batch_output(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn(fn=TestAverageFn(batch_output=False))
    self.assertEqual(2.0, agg_fn(data))

  def test_tree_aggregate_fn_tuple_output(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn(
        fn=TestAverageFn(batch_output=False, return_tuple=True),
        output_keys=Key.SELF,
    )
    self.assertEqual((2.0, 0), agg_fn(data))

  def test_tree_aggregate_fn_dict_output(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn(
        fn=TestAverageFnDictOutput(batch_output=False),
        output_keys={Key.SELF: 'mean'},
    )
    self.assertEqual(2.0, agg_fn(data))

  def test_tree_aggregate_fn_dict_multiple_output(self):
    data = [1, 2, 3]
    agg_fn = tree_fns.TreeAggregateFn(
        fn=TestAverageFnDictOutput(batch_output=False),
        output_keys=dict(output_mean='mean', state='state'),
    )
    self.assertEqual(dict(output_mean=2.0, state=(6, 3)), agg_fn(data))

  def test_tree_aggregate_with_lazyfn(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    agg_fn = tree_fns.TreeAggregateFn(
        input_keys='a', fn=lazy_fns.trace(TestAverageFn)(), output_keys='mean'
    )
    self.assertEqual({'mean': [2.0]}, agg_fn(data))

  def test_tree_aggregate_with_as_agg_fn(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    agg_fn = tree_fns.TreeAggregateFn(
        input_keys='a', fn=TestAverageMetric(), output_keys='mean'
    )
    self.assertEqual({'mean': [2.0]}, agg_fn(data))

  def test_tree_aggregate_with_lasy_as_agg_fn(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    agg_fn = tree_fns.TreeAggregateFn(
        input_keys='a',
        fn=lazy_fns.trace(TestAverageMetric)(),
        output_keys='mean',
    )
    self.assertEqual({'mean': [2.0]}, agg_fn(data))

  def test_tree_aggregate_invalid_agg_fn(self):
    with self.assertRaisesRegex(TypeError, 'Not an aggregatable'):
      tree_fns.TreeAggregateFn(fn=1)  # pytype: disable=wrong-arg-types

  def test_tree_aggregate_with_keyword_arg(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    bias = 1
    agg_fn = tree_fns.TreeAggregateFn(
        input_keys=dict(inputs='a', default_value=Key.Literal(bias)),
        fn=lazy_fns.trace(TestAverageFn)(),
        output_keys='mean',
    )
    expected_mean = [(sum(data['a']) + bias) / len(data['a'])]
    self.assertEqual({'mean': expected_mean}, agg_fn(data))

  def test_tree_aggregate_with_incorrect_keyword_arg(self):
    data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    agg_fn = tree_fns.TreeAggregateFn(
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
    tree_fn = tree_fns.TreeFn(
        input_keys=input_keys,
        masks=masks,
        replace_mask_false_with=replace_mask_false_with,
    )
    result = tree_fn(inputs)
    test_utils.assert_nested_container_equal(self, result, expected)

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
    tree_fn = tree_fns.TreeFn(
        input_keys=input_keys,
        masks=masks,
        replace_mask_false_with=replace_mask_false_with,
    )
    result = tree_fn(inputs)
    test_utils.assert_nested_container_equal(self, result, expected)


if __name__ == '__main__':
  absltest.main()
