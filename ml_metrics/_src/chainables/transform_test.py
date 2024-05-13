# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for transform."""
import functools
from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns

Key = tree.Key
MetricKey = transform.MetricKey
SliceKey = tree_fns.SliceKey


# TODO: b/318463291 - Improves test coverage.
class TestAverageFn:

  def __init__(self, batch_output=True, return_tuple=False):
    self.batch_output = batch_output
    self.return_tuple = return_tuple

  def create_state(self):
    return [0, 0]

  def update_state(self, state, inputs):
    return state[0] + sum(inputs), state[1] + len(inputs)

  def merge_states(self, states):
    result = [0, 0]
    for state in states:
      result[0] += state[0]
      result[1] += state[1]
    return result

  def get_result(self, state):
    result = state[0] / state[1]
    result = [result] if self.batch_output else result
    return (result, 0) if self.return_tuple else result


class TransformTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='apply_elem', inputs=0, fn=lambda x: x + 1, expected=1
      ),
      dict(
          testcase_name='apply_batch',
          inputs=[0, 1],
          fn=lazy_fns.iterate_fn(lambda x: x + 1),
          expected=[1, 2],
      ),
      dict(
          testcase_name='apply_with_input_keys',
          inputs=[0, 1],
          fn=lambda x: x + 1,
          expected=2,
          input_keys=Key.Index(1),
      ),
      dict(
          testcase_name='apply_with_output_keys',
          inputs=[0, 1],
          fn=lambda x: x + 1,
          expected={'output': 2},
          input_keys=Key.Index(1),
          output_keys='output',
      ),
      dict(
          testcase_name='apply_with_lazy_fn',
          inputs={'inputs': [0, 1]},
          fn=lazy_fns.trace(functools.partial)(len),
          expected={'output': 2},
          input_keys='inputs',
          output_keys='output',
      ),
  ])
  def test_apply_transform(
      self,
      inputs,
      fn,
      expected,
      input_keys=tree.Key.SELF,
      output_keys=tree.Key.SELF,
  ):
    t = transform.TreeTransform.new().apply(
        fn=fn, input_keys=input_keys, output_keys=output_keys
    )
    self.assertEqual(expected, t.make()(inputs))

  @parameterized.named_parameters([
      dict(
          testcase_name='assign_elem', inputs=0, fn=lambda x: x + 1, expected=1
      ),
      dict(
          testcase_name='assign_batch',
          inputs=[0, 1],
          fn=lazy_fns.iterate_fn(lambda x: x + 1),
          expected=[1, 2],
      ),
      dict(
          testcase_name='assign_with_input_keys',
          inputs=[0, 1],
          fn=lambda x: x + 1,
          expected=2,
          input_keys=Key.Index(1),
      ),
      dict(
          testcase_name='assign_with_output_keys',
          inputs=[0, 1],
          fn=lambda x: x + 1,
          expected=[0, 2],
          input_keys=Key.Index(1),
          output_keys=Key.Index(1),
      ),
      dict(
          testcase_name='assign_with_lazy_fn',
          inputs={'inputs': [0, 1]},
          fn=lazy_fns.trace(functools.partial)(len),
          expected={'inputs': [0, 1], 'output': 2},
          input_keys='inputs',
          output_keys='output',
      ),
  ])
  def test_assign_transform(
      self,
      inputs,
      fn,
      expected,
      input_keys=tree.Key.SELF,
      output_keys=tree.Key.SELF,
  ):
    t = transform.TreeTransform.new().assign(
        output_keys, fn=fn, input_keys=input_keys
    )
    self.assertEqual(expected, t.make()(inputs))

  def test_assign_invalid_keys(self):
    with self.assertRaises(ValueError):
      transform.TreeTransform.new().assign(fn=len).assign('a', fn=len)

    with self.assertRaises(ValueError):
      transform.TreeTransform.new().assign('a', fn=len).assign(fn=len)

  def test_assign_duplicate_keys(self):
    with self.assertRaises(ValueError):
      (
          transform.TreeTransform.new()
          .assign(('a', 'b'), fn=len)
          .assign('a', fn=len)
      )

  @parameterized.named_parameters([
      dict(testcase_name='select_self', inputs=0, expected=0),
      dict(
          testcase_name='select_index',
          inputs=[0, 1],
          input_keys=Key.Index(1),
          expected=1,
      ),
      dict(
          testcase_name='select_with_keys',
          inputs={'a': [0, 1], 'b': 1},
          expected=[0, 1],
          input_keys='a',
      ),
      dict(
          testcase_name='select_with_output_keys',
          inputs={'a': 0, 'b': 1},
          expected={'c': 1},
          input_keys='b',
          output_keys='c',
      ),
  ])
  def test_select_transform(
      self,
      inputs,
      expected,
      input_keys=Key.SELF,
      output_keys=Key.SELF,
  ):
    t = transform.TreeTransform.new().select(
        input_keys,
        output_keys=output_keys,
    )
    self.assertEqual(expected, t.make()(inputs))

  @parameterized.named_parameters([
      dict(
          testcase_name='agg_self',
          inputs=[0, 1, 2],
          fn=TestAverageFn(),
          expected=[1],
      ),
      dict(
          testcase_name='agg_with_input_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          fn=TestAverageFn(),
          expected=[2],
      ),
      dict(
          testcase_name='agg_with_output_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          output_keys='a',
          fn=TestAverageFn(),
          expected={'a': [2]},
      ),
  ])
  def test_aggregate_transform(
      self,
      inputs,
      fn,
      expected,
      input_keys=Key.SELF,
      output_keys=Key.SELF,
  ):
    t = transform.TreeTransform.new().aggregate(
        input_keys=input_keys,
        fn=fn,
        output_keys=output_keys,
    )
    self.assertEqual(expected, t.make()(inputs))

  def test_aggregate_invalid_keys(self):
    with self.assertRaises(ValueError):
      transform.TreeTransform.new().aggregate(
          fn=TestAverageFn(),
      ).add_aggregate(fn=TestAverageFn(), output_keys='a')

    with self.assertRaises(ValueError):
      transform.TreeTransform.new().aggregate(
          fn=TestAverageFn(),
          output_keys='a',
      ).add_aggregate(fn=TestAverageFn())

  def test_aggregate_duplicate_keys(self):
    with self.assertRaises(ValueError):
      transform.TreeTransform.new().aggregate(
          fn=TestAverageFn(),
          output_keys='a',
      ).add_aggregate(fn=TestAverageFn(), output_keys=('a', 'b'))

  @parameterized.named_parameters([
      dict(
          testcase_name='agg_self',
          inputs=[0, 1, 2],
          fn=lazy_fns.trace(TestAverageFn)(),
          expected=[1],
      ),
      dict(
          testcase_name='agg_with_input_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          fn=lazy_fns.trace(TestAverageFn)(),
          expected=[2],
      ),
      dict(
          testcase_name='agg_with_output_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          output_keys='a',
          fn=lazy_fns.trace(TestAverageFn)(),
          expected={'a': [2]},
      ),
      dict(
          testcase_name='agg_with_fn_instance',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          output_keys='a',
          fn=TestAverageFn(),
          expected={'a': [2]},
      ),
  ])
  def test_aggregate_transform_with_lazy_fn(
      self,
      inputs,
      fn,
      expected,
      input_keys=Key.SELF,
      output_keys=Key.SELF,
  ):
    t1 = transform.TreeTransform.new().aggregate(
        input_keys=input_keys,
        fn=fn,
        output_keys=output_keys,
    )
    t2 = transform.TreeTransform.new().aggregate(
        input_keys=input_keys,
        fn=fn,
        output_keys=output_keys,
    )
    agg1, agg2 = t1.make(), t2.make()
    state1 = agg1._update_state(agg1.create_state(), inputs)
    state2 = agg2._update_state(agg2.create_state(), inputs)
    # LazyFn of the fn enables across workers merge since these are consistent
    # after reinstantion of the actual function instance.
    merged_t1 = agg1.merge_states([state1, state2])
    merged_t2 = agg2.merge_states([state1, state2])
    self.assertEqual(merged_t1, merged_t2)
    self.assertEqual(expected, agg1.get_result(state1))
    self.assertEqual(expected, agg2.get_result(state2))

  def test_aggregate_with_slices(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 9, 5]}
    agg = (
        transform.TreeTransform.new()
        .aggregate(
            input_keys='b',
            fn=TestAverageFn(),
            output_keys='avg_b',
        )
        .add_slice('a')
        .add_slice('b')
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a',), (1,))): [3.0],
        MetricKey('avg_b', SliceKey(('a',), (2,))): [9.0],
        MetricKey('avg_b', SliceKey(('b',), (1,))): [1.0],
        MetricKey('avg_b', SliceKey(('b',), (9,))): [9.0],
        MetricKey('avg_b', SliceKey(('b',), (5,))): [5.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_with_slice_crosses(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 9, 5]}
    agg = (
        transform.TreeTransform.new()
        .aggregate(
            input_keys='b',
            fn=TestAverageFn(),
            output_keys='avg_b',
        )
        .add_slice(('a', 'b'))
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a', 'b'), (1, 1))): [1.0],
        MetricKey('avg_b', SliceKey(('a', 'b'), (2, 9))): [9.0],
        MetricKey('avg_b', SliceKey(('a', 'b'), (1, 5))): [5.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_with_slice_fn_fanout(self):
    inputs = {'a': ['x1 x2', 'x2', 'x1 x3'], 'b': [1, 9, 5]}

    def foo(x):
      return x.split(' ')

    agg = (
        transform.TreeTransform.new()
        .aggregate(
            input_keys='b',
            fn=TestAverageFn(),
            output_keys='avg_b',
        )
        .add_slice('a', slice_fn=foo)
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a',), ('x1',))): [3.0],
        MetricKey('avg_b', SliceKey(('a',), ('x2',))): [5.0],
        MetricKey('avg_b', SliceKey(('a',), ('x3',))): [5.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_with_slice_fn_multi_fanout(self):
    inputs = {'a': ['x1 x2', 'x2', 'x1 x3'], 'b': [1, 9, 5]}

    def foo(x):
      return ((i, x) for i, x in enumerate(x.split(' ')))

    agg = (
        transform.TreeTransform.new()
        .aggregate(
            input_keys='b',
            fn=TestAverageFn(),
            output_keys='avg_b',
        )
        .add_slice('a', slice_fn=foo, slice_name=('pos', 'token'))
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('pos', 'token'), (0, 'x1'))): [3.0],
        MetricKey('avg_b', SliceKey(('pos', 'token'), (0, 'x2'))): [9.0],
        MetricKey('avg_b', SliceKey(('pos', 'token'), (1, 'x2'))): [1.0],
        MetricKey('avg_b', SliceKey(('pos', 'token'), (1, 'x3'))): [5.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_raises_when_slice_fn_emits_unhashable(self):
    inputs = {'a': ['x1 x2', 'x2', 'x1 x3'], 'b': [1, 9, 5]}

    def foo(x):
      return ([i, x] for i, x in enumerate(x.split(' ')))

    agg = (
        transform.TreeTransform.new()
        .aggregate(
            input_keys='b',
            fn=TestAverageFn(),
            output_keys='avg_b',
        )
        .add_slice('a', slice_fn=foo, slice_name=('pos', 'a'))
    )
    with self.assertRaises(ValueError):
      agg.make()(inputs)

  def test_input_iterator_transform(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform.new()
        .data_source(iterator=input_iterator)
        .apply(fn=sum)
    )
    actual_fn = t.make()
    self.assertEqual([6, 9], list(actual_fn.iterate()))
    self.assertEqual([6, 9], list(actual_fn.iterate(input_iterator)))
    self.assertEqual([6, 9], actual_fn(input_iterator=input_iterator))

  def test_input_iterator_aggregate(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform.new()
        .data_source(iterator=lazy_fns.trace(lambda: input_iterator)())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
        .aggregate(fn=TestAverageFn())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 10))
        .aggregate(fn=TestAverageFn())
    )
    actual_fn: transform.CombinedTreeFn = t.make()
    self.assertEqual([13.5], actual_fn())
    self.assertEqual([13.5], actual_fn(input_iterator=input_iterator))

    state = actual_fn.merge_states(
        actual_fn.iterate(with_agg_state=True), mixed_input_types=True
    )
    self.assertEqual([13.5], actual_fn.get_result(state))

    state = actual_fn.merge_states(
        actual_fn.iterate(input_iterator=input_iterator, with_agg_state=True),
        mixed_input_types=True,
    )
    self.assertEqual([13.5], actual_fn.get_result(state))

  def test_input_iterator_aggregate_incorrect_states_count_raises_error(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform.new()
        .data_source(iterator=lazy_fns.trace(lambda: input_iterator)())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
        .aggregate(fn=TestAverageFn())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 10))
        .aggregate(fn=TestAverageFn())
    )
    actual_fn: transform.CombinedTreeFn = t.make()
    with self.assertRaises(ValueError):
      actual_fn.merge_states(
          actual_fn.iterate(with_agg_state=True),
          mixed_input_types=True,
          strict_states_cnt=2,
      )

  def test_chain(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    read = transform.TreeTransform.new().data_source(
        iterator=lazy_fns.trace(lambda: input_iterator)()
    )
    process = (
        transform.TreeTransform.new()
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
        .aggregate(fn=TestAverageFn())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 10))
        .aggregate(fn=TestAverageFn())
    )
    t = read.chain(process)
    actual_fn = t.make()
    self.assertEqual([13.5], actual_fn())
    self.assertEqual([13.5], actual_fn(input_iterator=input_iterator))

  def test_prefetched_iterator(self):
    input_iterator = range(10)
    p = transform.TreeTransform.new().data_source(iterator=input_iterator)
    iterator = transform.PrefetchableIterator(
        p.make().iterate(), prefetch_size=2
    )
    iterator.prefetch()
    self.assertEqual(2, iterator.cnt)
    self.assertEqual(list(range(10)), list(iterator))


if __name__ == '__main__':
  absltest.main()
