# Copyright 2025 Google LLC
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

from collections.abc import Callable
from concurrent import futures
import dataclasses
import functools
import itertools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.chainables import io
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns
from ml_metrics._src.utils import iter_utils
from ml_metrics._src.utils import test_utils
import more_itertools as mit
import numpy as np


Key = tree.Key
MetricKey = transform.MetricKey
SliceKey = tree_fns.SliceKey


class BatchedCall:

  def __init__(self, batch_size):
    self._batch_size = batch_size

  def __call__(self, x):
    assert len(x) <= self._batch_size
    return np.array(x) + 1


def _reduce_sum(inputs):
  if isinstance(inputs, bool):
    return int(inputs)
  return sum(_reduce_sum(elem) for elem in inputs)


def _reduce_size(inputs):
  if isinstance(inputs, (tuple, list)):
    result = sum(_reduce_size(elem) for elem in inputs)
  else:
    result = 1
  return result


def _get_all_threads(iterators: list[transform._RunnerIterator]):
  it_theads = (p._threads for it in iterators if (p := it._thread_pool))
  return list(itertools.chain.from_iterable(it_theads))


# TODO: b/318463291 - Improves test coverage.
class MockAverageFn:

  def __init__(
      self,
      batch_output=True,
      return_tuple=False,
      input_key=None,
      return_dict_key='',
  ):
    self.batch_output = batch_output
    self.return_tuple = return_tuple
    self.return_dict = return_dict_key
    self.input_key = input_key

  def create_state(self):
    return [0, 0]

  def update_state(self, state, inputs):
    inputs = inputs[self.input_key] if self.input_key else inputs
    if isinstance(inputs, (int, float)):
      return state[0] + inputs, state[1] + 1
    return state[0] + sum(inputs), state[1] + len(inputs)

  def merge_states(self, states):
    result = [0, 0]
    for state in states:
      result[0] += state[0]
      result[1] += state[1]
    return result

  def get_result(self, state):
    result = state[0] / state[1] if state[1] else np.nan
    result = [result] if self.batch_output else result
    if self.return_dict:
      return {self.return_dict: result}
    return (result, 0) if self.return_tuple else result


@dataclasses.dataclass
class MockPrecisionRecall:
  matcher: Callable[[Any, Any], Any] | None = None
  pred_tp: int = 0
  gp_tp: int = 0
  pred_p: int = 0
  gt_p: int = 0

  def add(self, pred=None, label=None, matched_pred=None, matched_label=None):
    # The matcher can be bypassed if the matched results are provided. This
    # is to test for explicit matcher and implicit matcher logic when
    # interacting with intra-example slicing.
    if matched_pred is None or matched_label is None:
      matched_pred, matched_label = self.matcher(pred, label)
    self.pred_tp += _reduce_sum(matched_pred)
    self.gp_tp += _reduce_sum(matched_label)
    self.pred_p += _reduce_size(matched_pred)
    self.gt_p += _reduce_size(matched_label)

  def precision(self):
    return self.pred_tp / self.pred_p if self.pred_p else float('nan')

  def recall(self):
    return self.gp_tp / self.gt_p if self.gt_p else float('nan')

  def result(self):
    return self.precision(), self.recall()


def get_mask(inputs, key):
  if isinstance(inputs, list):
    return [get_mask(elem, key) for elem in inputs]
  else:
    return inputs == key


def single_slicer(inputs, for_pred=True, within=()):
  keys = set()
  for elems in inputs:
    for elem in elems:
      keys.add(elem)
  within = within or keys
  for key in sorted(keys):
    if key in within:
      mask = get_mask(inputs, key)
      masks = (mask, True) if for_pred else (True, mask)
      yield key, masks


def multi_slicer(preds, labels, within=()):
  keys = set()
  for elem in itertools.chain(preds, labels):
    keys.update(elem)
  within = within or keys
  for key in sorted(keys):
    if key in within:
      pred_mask = get_mask(preds, key)
      label_mask = get_mask(labels, key)
      masks = (pred_mask, label_mask)
      yield key, masks


class TransformDataSourceTest(parameterized.TestCase):

  def test_sharded_sequence_data_source_make(self):
    ds = io.SequenceDataSource(range(3))
    p = transform.TreeTransform().data_source(ds).apply(lambda x: x + 1)
    num_shards = 2
    shards = (io.ShardConfig(i, num_shards) for i in range(num_shards))
    actual = [list(p.make(shard=shard)) for shard in shards]
    expected = [[1, 2], [3]]
    self.assertEqual(expected, actual)

  def test_sharded_sequence_data_source_multithread(self):
    ds = io.SequenceDataSource(range(512))
    num_threads = 64
    p = transform.TreeTransform(num_threads=num_threads).data_source(ds)
    expected = list(range(512))
    it = p.make().iterate()
    self.assertCountEqual(expected, list(it))
    threads = _get_all_threads(it._iterators)
    self.assertNotEmpty(threads)
    self.assertTrue(all(not t.is_alive() for t in threads))

  def test_sharded_sequence_data_source_resume(self):
    ds = io.SequenceDataSource(range(3))
    p = (
        transform.TreeTransform()
        .data_source(ds)
        .apply(lambda x: x + 1)
        .agg(MockAverageFn())
    )
    num_shards = 2
    it = p.make(shard=io.ShardConfig(0, num_shards)).iterate()
    self.assertEqual(1, next(it))
    self.assertEqual({'': [1]}, it.agg_result)
    # Recover from the iterator state, the results of the original and the new
    # iterator should be the same.
    it_new = p.make().iterate().from_state(it.state)
    self.assertEqual([2], list(it))
    self.assertEqual([2], list(it_new))
    self.assertEqual({'': [(1 + 2) / 2]}, it.agg_result)
    self.assertEqual({'': [(1 + 2) / 2]}, it_new.agg_result)

  def test_sequence_data_source_apply_ignore_error(self):
    def foo(x):
      if x == 2:
        raise ValueError('foo')
      return x

    num_threads = 1
    p = transform.TreeTransform(num_threads=num_threads).apply(foo)
    expected = [0, 1, 3]
    it = p.make().iterate(range(4), ignore_error=True)
    self.assertEqual(expected, list(it))
    threads = _get_all_threads(it._iterators)
    self.assertNotEmpty(threads)
    self.assertTrue(all(not t.is_alive() for t in threads))

  def test_sequence_data_source_assign_ignore_error(self):
    def foo(x):
      if x == 2:
        raise ValueError('foo')
      return x

    num_threads = 1
    p = (
        transform.TreeTransform(num_threads=num_threads)
        .apply(lambda x: x, output_keys='a')
        .assign('b', fn=foo, input_keys='a')
    )
    expected = [{'a': 0, 'b': 0}, {'a': 1, 'b': 1}, {'a': 3, 'b': 3}]
    it = p.make().iterate(range(4), ignore_error=True)
    self.assertEqual(expected, list(it))
    threads = _get_all_threads(it._iterators)
    self.assertNotEmpty(threads)
    self.assertTrue(all(not t.is_alive() for t in threads))

  def test_mock_generator_bool_operator(self):
    ds = test_utils.NoLenIter(range(3))
    with self.assertRaisesRegex(ValueError, 'Cannot call len()'):
      if ds:
        pass

  def test_data_source_not_recoverable_raise_error(self):
    ds = iter(range(3))
    p = transform.TreeTransform().data_source(ds).apply(lambda x: x + 1)
    with self.assertRaisesRegex(
        TypeError, 'Data source is not serializable, got .+'
    ):
      _ = p.make().iterate().state
    with self.assertRaisesRegex(
        TypeError, 'Data source is not recoverable, got .+'
    ):
      p.make().iterate().from_state(
          transform._IteratorState([io.ShardConfig()], agg_state=None)
      )

  def test_input_override(self):
    p = transform.TreeTransform().data_source(range(3)).apply(lambda x: x + 1)
    runner = transform.TransformRunner.from_transform(p, input_state=None)
    self.assertEqual(list(runner.iterate(range(3))), [1, 2, 3])

  def test_sharded_iterable_data_source(self):
    ds = io.ShardedIterable(range(3))
    p = transform.TreeTransform().data_source(ds).apply(lambda x: x + 1)
    num_shards = 2
    shards = (io.ShardConfig(i, num_shards) for i in range(num_shards))
    actual = [list(p.make(shard=shard)) for shard in shards]
    expected = [[1, 3], [2]]
    self.assertEqual(expected, actual)

  def test_nonshardable_data_source_with_shard_index_raise_error(self):
    ds = test_utils.NoLenIter(range(3))
    p = transform.TreeTransform().data_source(ds).apply(lambda x: x + 1)
    with self.assertRaisesRegex(
        TypeError, 'Data source is not configurable but .+ is provided.'
    ):
      _ = list(p.make(shard=io.ShardConfig(shard_index=1)))

  def test_data_source_size(self):
    p = transform.TreeTransform().data_source(range(100)).apply(lambda x: x + 1)
    it = p.make().iterate(data_source_size=transform.AUTO_SIZE)
    self.assertLen(it, 100)


class TransformTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.maxDiff = None

  def test_transform_unique_ids_with_each_op(self):
    seen_ids = set()
    t = transform.TreeTransform()
    seen_ids.add(t.id)
    t = dataclasses.replace(t, name='foo')
    seen_ids.add(t.id)
    t = t.apply(output_keys='a', fn=lambda x: x + 1)
    seen_ids.add(t.id)
    t = t.assign('b', fn=lambda x: x + 1)
    seen_ids.add(t.id)
    t = dataclasses.replace(t, name='')
    t = t.aggregate(MockAverageFn(), output_keys='c')
    seen_ids.add(t.id)
    t = t.add_aggregate(MockAverageFn(), output_keys='d').add_slice('a')
    seen_ids.add(t.id)
    # Each new transform should create a unique id.
    self.assertLen(seen_ids, 6)

  def test_transform_name(self):
    t = (
        transform.TreeTransform.new(name='call')
        .data_source(test_utils.NoLenIter(range(3)))
        .apply(lambda x: x + 1)
    )
    self.assertEqual(t.make()._runners[0].name, 'call')

  def test_transform_call(self):
    t = (
        transform.TreeTransform.new(name='call')
        .data_source(test_utils.NoLenIter(range(3)))
        .apply(lambda x: x + 1)
    )
    self.assertEqual(list(t.make()), [1, 2, 3])
    self.assertEqual([2, 3, 4], list(t.make().iterate(range(1, 4))))
    self.assertEqual(2, t.make()(1))

  @parameterized.named_parameters([
      dict(
          testcase_name='apply',
          transform_fn=lambda p: p.apply(len, output_keys='b'),
          expected={'b'},
      ),
      dict(
          testcase_name='assign',
          transform_fn=lambda p: p.assign('b', fn=len),
          expected={'a', 'b'},
      ),
      dict(
          testcase_name='assign_apply',
          transform_fn=lambda p: p.assign('b', fn=len).apply(output_keys='c'),
          expected={'c'},
      ),
      dict(
          testcase_name='assign_select',
          transform_fn=lambda p: p.assign('b', fn=len).select('b'),
          expected={'b'},
      ),
      dict(
          testcase_name='apply_select',
          transform_fn=lambda p: p.apply(output_keys='b', fn=len).select('b'),
          expected={'b'},
      ),
      dict(
          testcase_name='filter',
          transform_fn=lambda p: p.filter(lambda x: x > 0, input_keys='a'),
          expected={'a'},
      ),
      dict(
          testcase_name='batch',
          transform_fn=lambda p: p.batch(2),
          expected={'a'},
      ),
      dict(
          testcase_name='assign_batch',
          transform_fn=lambda p: p.assign('b', fn=len).batch(2),
          expected={'a', 'b'},
      ),
      dict(
          testcase_name='assign_filter',
          transform_fn=lambda p: p.assign('b').filter(len, input_keys='a'),
          expected={'a', 'b'},
      ),
  ])
  def test_output_keys(self, transform_fn, expected):
    p = transform_fn(transform.TreeTransform().apply(len, output_keys='a'))
    self.assertEqual(p.output_keys, expected)

  def test_output_batch_size(self):
    p = (
        transform.TreeTransform()
        .apply(lambda x: x + 1, output_keys='a')
        .batch(2)
        .assign('b', fn=lambda x: x + 1, input_keys='a')
    )
    self.assertEqual(p.batch_size, 2)
    p = p.apply(lambda x: x + 1)
    self.assertEqual(p.batch_size, 2)
    p = p.batch(3, batch_fn=lambda x: x).agg(MockAverageFn(), output_keys='c')
    self.assertEqual(p.batch_size, 3)

  def test_output_keys_aggregate(self):
    p = (
        transform.TreeTransform()
        .apply(len, output_keys='a')
        .apply(len, output_keys='b')
        .assign('c', fn=lambda x: x + 1, input_keys='b')
    )
    p = p.agg(MockAverageFn(), output_keys='c').add_aggregate(
        MockAverageFn(),
        output_keys='d',
    )
    self.assertEqual(p.output_keys, {'c', 'b'})
    self.assertEqual(p.agg_output_keys, {'c', 'd'})

  def test_non_aggregate_call_with_iterator_raise_error(self):
    t = transform.TreeTransform().data_source(test_utils.NoLenIter(range(3)))
    with self.assertRaisesRegex(
        ValueError, 'Non-aggregate transform is not callable with iterator'
    ):
      _ = list(t.make()())

    t = transform.TreeTransform().apply(lambda x: x + 1)
    with self.assertRaisesRegex(
        ValueError, 'Non-aggregate transform is not callable with iterator'
    ):
      _ = list(t.make()(input_iterator=range(3)))

  def test_transform_named_transforms_default(self):
    t1 = (
        transform.TreeTransform()
        .data_source(test_utils.NoLenIter(range(3)))
        .apply(lambda x: x + 1)
    )
    t2 = (
        transform.TreeTransform.new(name='A')
        .apply(iter_utils.iterate_fn(lambda x: x + 1))
        .agg(lazy_fns.trace(MockAverageFn)())
    )
    t3 = transform.TreeTransform.new(name='B').apply(
        iter_utils.iterate_fn(lambda x: x + 1)
    )
    t = t1.interleave(t2).interleave(t3)
    self.assertSameElements(t.named_transforms().keys(), ('', 'A', 'B'))
    # The first group of nodes are identical.
    self.assertEqual(t.named_transforms()[''], t1)
    self.assertEqual(list(t.named_transforms()[''].make()), list(t1.make()))
    inputs = [1, 2, 3]
    self.assertEqual(
        t.named_transforms()['A'].make()(inputs), t2.make()(inputs)
    )
    inputs = [1, 2, 3]
    self.assertEqual(
        list(t.named_transforms()['B'].make().iterate(inputs)),
        list(t3.make().iterate(inputs)),
    )

  def test_transform_filter_with_input_keys(self):
    t = (
        transform.TreeTransform()
        .data_source(range(5))
        .apply(lambda x: (x, x + 1), output_keys=('a', 'b'))
        .filter(lambda x: x % 2 == 0, input_keys='a')
        .batch(batch_size=2)
    )
    self.assertEqual(
        list(t.make()), [{'a': [0, 2], 'b': [1, 3]}, {'a': [4], 'b': [5]}]
    )

  def test_transform_filter_with_self(self):
    t = (
        transform.TreeTransform()
        .data_source(range(5))
        .filter(lambda x: x % 2 == 0)
        .batch(batch_size=2)
    )
    self.assertEqual(list(t.make()), [[0, 2], [4]])

  def test_transform_named_transforms_with_duplicate_names(self):
    t1 = transform.TreeTransform.new(name='A').data_source(
        test_utils.NoLenIter(range(3))
    )
    t2 = transform.TreeTransform.new(name='B').agg(
        lazy_fns.trace(MockAverageFn)()
    )
    t3 = transform.TreeTransform.new(name='A').apply(
        fn=iter_utils.iterate_fn(lambda x: x + 1)
    )
    with self.assertRaisesRegex(ValueError, 'Chaining duplicate transform'):
      _ = t1.interleave(t2).interleave(t3)

  def test_transform_named_transforms_with_duplicate_agg_keys(self):
    t1 = transform.TreeTransform.new(name='A').data_source(
        test_utils.NoLenIter(range(3))
    )
    t2 = transform.TreeTransform.new(name='B').agg(
        lazy_fns.trace(MockAverageFn)()
    )
    t3 = (
        transform.TreeTransform.new(name='C')
        .apply(fn=iter_utils.iterate_fn(lambda x: x + 1))
        .agg(lazy_fns.trace(MockAverageFn)())
    )
    with self.assertRaisesRegex(
        ValueError, 'Chaining transforms with duplicate aggregation output keys'
    ):
      _ = t1.interleave(t2).interleave(t3)

  def test_transform_equal(self):
    t = transform.TreeTransform()
    pickled_t = lazy_fns.pickler.dumps(lazy_fns.trace(t))
    self.assertEqual(
        lazy_fns.maybe_make(pickled_t),
        lazy_fns.maybe_make(pickled_t),
    )

  def test_transform_iterate_is_not_copy(self):
    t = (
        transform.TreeTransform()
        .data_source(test_utils.inf_range(2))
        .apply(lambda x: x + 1, output_keys='a')
        .assign('b', fn=lambda x: x + 1, input_keys='a')
    )
    actual = list(itertools.islice(t.make(), 2))
    self.assertEqual(actual, [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}])

  @parameterized.named_parameters([
      dict(
          testcase_name='apply_elem', inputs=0, fn=lambda x: x + 1, expected=1
      ),
      dict(
          testcase_name='apply_batch',
          inputs=[0, 1],
          fn=iter_utils.iterate_fn(lambda x: x + 1),
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
      dict(
          testcase_name='with_kwargs_input_by_iterate_fn',
          inputs={'a': [0, 1], 'b': [7, 8]},
          fn=iter_utils.iterate_fn(lambda x, y: (x + 1, y + 2)),
          expected=((1, 2), (9, 10)),
          input_keys=dict(x='a', y='b'),
      ),
      dict(
          testcase_name='with_kwargs_input',
          inputs={'a': 0, 'b': 1},
          fn=lambda x, y: (x + 1, y + 2),
          expected=(1, 3),
          input_keys=dict(x='a', y='b'),
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
    t = transform.TreeTransform().apply(
        fn=fn, input_keys=input_keys, output_keys=output_keys
    )
    self.assertEqual(expected, t.make()(inputs))

  @parameterized.named_parameters([
      dict(
          testcase_name='assign_elem',
          inputs=[0],
          fn=lambda x: x[0] + 1,
          output_keys=Key().at(1),
          expected=[0, 1],
      ),
      dict(
          testcase_name='assign_batch',
          inputs=[[0, 1]],
          fn=iter_utils.iterate_fn(lambda x: x + 1),
          input_keys=Key().at(0),
          output_keys=Key().at(1),
          expected=[[0, 1], [1, 2]],
      ),
      dict(
          testcase_name='assign_with_output_keys',
          inputs=[0, 1],
          fn=lambda x: x + 1,
          expected=[0, 1, 2],
          input_keys=Key.Index(1),
          output_keys=Key.Index(2),
      ),
      dict(
          testcase_name='assign_with_lazy_fn',
          inputs={'inputs': [0, 1]},
          fn=lazy_fns.trace(functools.partial)(len),
          expected={'inputs': [0, 1], 'output': 2},
          input_keys='inputs',
          output_keys='output',
      ),
      dict(
          testcase_name='assign_with_dict_assign_keys',
          inputs={'inputs': [0, 1]},
          expected={'inputs': [0, 1], 'output': 1},
          output_keys=dict(output=Key.new('inputs', -1)),
      ),
  ])
  def test_assign_transform(
      self,
      inputs,
      expected,
      fn=None,
      input_keys=tree.Key.SELF,
      output_keys=tree.Key.SELF,
  ):
    t = transform.TreeTransform().assign(
        output_keys, fn=fn, input_keys=input_keys
    )
    self.assertEqual(expected, t.make()(inputs))

  @parameterized.named_parameters([
      dict(
          testcase_name='call',
          inputs=[{'a': [0, 1, 2], 'b': [1, 2, 3]}, {'a': [4, 5], 'b': [5, 6]}],
          fn_batch_size=2,
          batch_size=3,
          fn=lambda x, y: (BatchedCall(2)(x), BatchedCall(2)(y)),
          input_keys=('a', 'b'),
          output_keys=('c', 'd'),
          expected=[
              {'c': np.array([1, 2, 3]), 'd': np.array([2, 3, 4])},
              {'c': np.array([5, 6]), 'd': np.array([6, 7])},
          ],
      ),
      dict(
          testcase_name='select_only',
          inputs=[{'a': [0, 1, 2], 'b': [1, 2, 3]}, {'a': [3, 4], 'b': [4, 5]}],
          batch_size=2,
          input_keys=('a', 'b'),
          output_keys=('c', 'd'),
          expected=[
              {'c': [0, 1], 'd': [1, 2]},
              {'c': [2, 3], 'd': [3, 4]},
              {'c': [4], 'd': [5]},
          ],
      ),
      dict(
          testcase_name='output_keys_only',
          inputs=map(list, mit.batched(range(5), 3)),
          fn_batch_size=2,
          batch_size=3,
          output_keys='a',
          fn=BatchedCall(batch_size=2),
          expected=[{'a': np.array([1, 2, 3])}, {'a': np.array([4, 5])}],
      ),
      dict(
          testcase_name='input_keys_only',
          inputs=[{'a': [0, 1, 2], 'b': [1]}, {'a': [4, 5], 'b': [5]}],
          fn_batch_size=2,
          batch_size=3,
          input_keys='a',
          fn=BatchedCall(batch_size=2),
          expected=[np.array([1, 2, 3]), np.array([5, 6])],
      ),
      dict(
          testcase_name='without_keys',
          inputs=map(list, mit.batched(range(5), 3)),
          fn_batch_size=2,
          batch_size=3,
          fn=BatchedCall(batch_size=2),
          expected=[np.array([1, 2, 3]), np.array([4, 5])],
      ),
      dict(
          testcase_name='with_lazy_fns',
          inputs=map(list, mit.batched(range(5), 3)),
          fn_batch_size=2,
          batch_size=3,
          fn=lazy_fns.trace(BatchedCall)(batch_size=2),
          expected=[np.array([1, 2, 3]), np.array([4, 5])],
      ),
      dict(
          testcase_name='rebatch_only',
          inputs=map(list, mit.batched(range(5), 2)),
          batch_size=3,
          expected=[[0, 1, 2], [3, 4]],
      ),
  ])
  def test_apply_transform_rebatched(
      self,
      inputs,
      expected,
      fn=None,
      fn_batch_size=0,
      batch_size=0,
      input_keys=tree.Key.SELF,
      output_keys=tree.Key.SELF,
  ):
    t = (
        transform.TreeTransform()
        .select(input_keys)
        # batch_fn is lambda x: x is equivalent to rebatching.
        .batch(fn_batch_size, batch_fn=lambda x: x)
        .apply(
            output_keys=output_keys,
            fn=fn,
            input_keys=input_keys,
        )
        .batch(batch_size=batch_size, batch_fn=lambda x: x)
    )
    actual = list(t.make().iterate(inputs))
    test_utils.assert_nested_container_equal(self, expected, actual)

  @parameterized.named_parameters([
      dict(
          testcase_name='call',
          inputs=[{'a': [0, 1, 2], 'b': [1, 2, 3]}, {'a': [4, 5], 'b': [5, 6]}],
          fn_batch_size=2,
          batch_size=3,
          fn=lambda x, y: (
              BatchedCall(2)(x).tolist(),
              BatchedCall(2)(y).tolist(),
          ),
          input_keys=('a', 'b'),
          output_keys=('c', 'd'),
          expected=[
              {'a': [0, 1, 2], 'b': [1, 2, 3], 'c': [1, 2, 3], 'd': [2, 3, 4]},
              {'a': [4, 5], 'b': [5, 6], 'c': [5, 6], 'd': [6, 7]},
          ],
      ),
      dict(
          testcase_name='call_single_input_single_output',
          inputs=({'a': list(batch)} for batch in mit.batched(range(5), 3)),
          fn_batch_size=2,
          batch_size=3,
          input_keys='a',
          output_keys='b',
          fn=BatchedCall(batch_size=2),
          expected=[
              {'a': [0, 1, 2], 'b': np.array([1, 2, 3])},
              {'a': [3, 4], 'b': np.array([4, 5])},
          ],
      ),
  ])
  def test_assign_transform_rebatched(
      self,
      inputs,
      expected,
      output_keys,
      fn=None,
      fn_batch_size=0,
      batch_size=0,
      input_keys=tree.Key.SELF,
  ):
    t = (
        transform.TreeTransform()
        .select(input_keys)
        # batch_fn is lambda x: x is equivalent to rebatching.
        .batch(fn_batch_size, batch_fn=lambda x: x)
        .assign(
            assign_keys=output_keys,
            fn=fn,
            input_keys=input_keys,
        )
        .batch(batch_size=batch_size, batch_fn=lambda x: x)
    )
    actual = list(t.make().iterate(inputs))
    test_utils.assert_nested_container_equal(self, expected, actual)

  def test_assign_invalid_keys(self):
    with self.assertRaises(ValueError):
      transform.TreeTransform().assign(fn=len).assign('a', fn=len)

    with self.assertRaises(ValueError):
      transform.TreeTransform().assign('a', fn=len).assign(fn=len)

  def test_deprecated_batch_size(self):
    with self.assertRaisesRegex(ValueError, 'batch_size is deprecated'):
      transform.TreeTransform().apply(batch_size=1)
    with self.assertRaisesRegex(ValueError, 'batch_size is deprecated'):
      transform.TreeTransform().apply(fn_batch_size=1)
    with self.assertRaisesRegex(ValueError, 'batch_size is deprecated'):
      transform.TreeTransform().assign('a', fn=len, batch_size=1)
    with self.assertRaisesRegex(ValueError, 'batch_size is deprecated'):
      transform.TreeTransform().assign('a', fn=len, fn_batch_size=1)
    with self.assertRaisesRegex(ValueError, 'batch_size is deprecated'):
      transform.TreeTransform().select(('a', 'b'), batch_size=1)

  @parameterized.named_parameters([
      dict(
          testcase_name='simple_keys',
          k1=('a', 'b'),
          k2='a',
      ),
      dict(
          testcase_name='dict_keys_last',
          k1=('a', 'b', 'c'),
          k2=dict(a='a', b='b'),
      ),
      dict(
          testcase_name='dict_keys_first',
          k1=dict(a='a', b='b'),
          k2=('a', 'b', 'c'),
      ),
      dict(
          testcase_name='tuple_of_dict_keys',
          k1=(dict(a='a', b='b'), 'c'),
          k2=('a', 'b', 'c'),
      ),
      dict(
          testcase_name='tuple_of_dict_keys_last',
          k1=('a', 'b', 'c'),
          k2=(dict(a='a', b='b'), 'c'),
      ),
  ])
  def test_assign_duplicate_keys_invalid(self, k1, k2):
    with self.assertRaisesRegex(KeyError, 'Duplicate output_keys'):
      # fn=len is not sensible here but the error should be raised at pipeline
      # construction time. So not reaching that point is part of the test.
      (transform.TreeTransform().assign(k1, fn=len).assign(k2, fn=len))

  @parameterized.named_parameters([
      dict(
          testcase_name='self_key',
          k1=Key.SELF,
          k2=('a', 'b', 'c'),
      ),
      dict(
          testcase_name='self_key_last',
          k1=('a', 'b', 'c'),
          k2=Key.SELF,
      ),
  ])
  def test_assign_self_keys_invalid(self, k1, k2):
    with self.assertRaisesRegex(KeyError, 'Cannot assign to SELF'):
      # fn=len is not sensible here but the error should be raised at pipeline
      # construction time. So not reaching that point is part of the test.
      (transform.TreeTransform().assign(k1, fn=len).assign(k2, fn=len))

  @parameterized.named_parameters([
      dict(testcase_name='select_self', inputs=[0], expected=[0]),
      dict(
          testcase_name='select_index',
          inputs=[0, 1],
          input_keys=Key.Index(1),
          expected=1,
      ),
      dict(
          testcase_name='select_with_a_key',
          inputs={'a': [0, 1], 'b': 1},
          input_keys='a',
          expected=[0, 1],
      ),
      dict(
          testcase_name='select_with_keys',
          inputs={'a': [0, 1], 'b': 1},
          input_keys=('a', 'b'),
          expected=([0, 1], 1),
      ),
      dict(
          testcase_name='select_with_output_keys',
          inputs={'a': 0, 'b': 1},
          input_keys=('b', 'a'),
          output_keys=('c', 'd'),
          expected={'c': 1, 'd': 0},
      ),
      dict(
          testcase_name='select_SELF_with_output_keys',
          inputs=(0, 1),
          output_keys='inputs',
          expected={'inputs': (0, 1)},
      ),
  ])
  def test_select_transform(
      self,
      inputs,
      expected,
      input_keys=Key.SELF,
      output_keys=Key.SELF,
  ):
    t = transform.TreeTransform().select(
        input_keys,
        output_keys=output_keys,
    )
    self.assertEqual(expected, t.make()(inputs))

  def test_select_with_rebatch(self):
    t = (
        transform.TreeTransform()
        .select('a', 'b')
        .batch(batch_size=3, batch_fn=lambda x: x)
    )
    inputs = [
        {'a': [0, 1], 'b': [1, 2], 'c': [0, 1]},
        {'a': [2, 3], 'b': [3, 4], 'c': [2, 3]},
    ]
    actual = list(t.make().iterate(inputs))
    expected = [
        {'a': [0, 1, 2], 'b': [1, 2, 3]},
        {'a': [3], 'b': [4]},
    ]
    self.assertEqual(expected, actual)

  def test_batch_direct(self):
    t = transform.TreeTransform.new().batch(batch_size=3)
    actual = list(t.make().iterate(range(10)))
    expected = list(map(list, mit.batched(range(10), 3)))
    self.assertEqual(expected, actual)

  def test_batch_with_apply(self):
    fn = lambda x: (x['a'], x['b'])
    inputs = [
        {'a': 0, 'b': 1},
        {'a': 2, 'b': 3},
        {'a': 4, 'b': 5},
    ]
    t = (
        transform.TreeTransform.new()
        .apply(fn, output_keys=('a', 'b'))
        .batch(batch_size=2)
    )
    actual = list(t.make().iterate(inputs))
    expected = [{'a': [0, 2], 'b': [1, 3]}, {'a': [4], 'b': [5]}]
    self.assertEqual(expected, actual)

  def test_batch_with_batch_with_select(self):
    inputs = [
        {'a': 0, 'b': 1},
        {'a': 2, 'b': 3},
        {'a': 4, 'b': 5},
    ]
    t = transform.TreeTransform.new().select('a', 'b').batch(batch_size=2)
    actual = list(t.make().iterate(inputs))
    expected = [{'a': [0, 2], 'b': [1, 3]}, {'a': [4], 'b': [5]}]
    self.assertEqual(expected, actual)

  def test_sink(self):
    sink = test_utils.TestSink()
    t = transform.TreeTransform().sink(sink, input_keys='a')
    inputs = [
        {'a': 0, 'b': 1},
        {'a': 2, 'b': 3},
        {'a': 4, 'b': 5},
    ]
    actual = list(t.make().iterate(inputs))
    self.assertEqual(inputs, actual)
    self.assertEqual([0, 2, 4], sink.data)

  def test_aggregate_starts_with_empty_agg(self):
    p = (
        transform.TreeTransform()
        .apply(lambda x: x * 10)
        .agg()
        .add_agg(MockAverageFn(), output_keys='a')
        .add_agg(MockAverageFn(), output_keys='b')
    )
    it_p = p.make().iterate([1, 2, 3])
    self.assertEqual([10, 20, 30], list(it_p))
    self.assertLen(p.fns, 1)
    self.assertLen(p.agg_fns, 2)
    self.assertEqual({'a': [20.0], 'b': [20.0]}, it_p.agg_result)

  def test_aggregate_empty_agg_raises_error(self):
    t = transform.TreeTransform().apply(lambda x: x * 10).agg()
    self.assertEmpty(t.make().agg_fns)

  def test_aggregate_not_last_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Aggregation has to be the last'):
      _ = transform.TreeTransform().agg(MockAverageFn()).apply(fn=lambda x: x)

  @parameterized.named_parameters([
      dict(
          testcase_name='agg_self',
          inputs=[0, 1, 2],
          fn=MockAverageFn(),
          expected=[1],
      ),
      dict(
          testcase_name='agg_with_input_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          fn=MockAverageFn(),
          expected=[2],
      ),
      dict(
          testcase_name='agg_with_output_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          output_keys='a',
          fn=MockAverageFn(),
          expected={'a': [2]},
      ),
      dict(
          testcase_name='agg_with_dict_output_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          output_keys=dict(c='avg'),
          fn=MockAverageFn(return_dict_key='avg'),
          expected={'c': [2]},
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
    t = transform.TreeTransform().aggregate(
        fn, input_keys=input_keys, output_keys=output_keys
    )
    self.assertEqual(expected, t.make()(inputs))

  def test_aggregate_invalid_keys(self):
    with self.assertRaisesRegex(KeyError, 'Cannot mix SELF with other keys'):
      transform.TreeTransform().agg(
          MockAverageFn(), output_keys=tree.Key.SELF
      ).add_aggregate(MockAverageFn(), output_keys='a')

    with self.assertRaisesRegex(KeyError, 'Cannot mix SELF with other keys'):
      transform.TreeTransform().aggregate(
          MockAverageFn(), output_keys='a'
      ).add_aggregate(MockAverageFn(), output_keys=tree.Key.SELF)

  def test_aggregate_duplicate_keys(self):
    with self.assertRaisesRegex(KeyError, 'Duplicate output_keys'):
      transform.TreeTransform().aggregate(
          MockAverageFn(), output_keys='a'
      ).add_aggregate(MockAverageFn(), output_keys=('a', 'b'))

    with self.assertRaisesRegex(KeyError, 'Duplicate output_keys'):
      transform.TreeTransform().aggregate(MockAverageFn()).add_aggregate(
          MockAverageFn()
      )

  def test_add_slice_without_agg_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'Cannot add slice without agg'):
      transform.TreeTransform().add_slice('a')

  @parameterized.named_parameters([
      dict(
          testcase_name='agg_self',
          inputs=[0, 1, 2],
          fn=lazy_fns.trace(MockAverageFn)(),
          expected=[1],
      ),
      dict(
          testcase_name='agg_with_input_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          fn=lazy_fns.trace(MockAverageFn)(),
          expected=[2],
      ),
      dict(
          testcase_name='agg_with_output_keys',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          output_keys='a',
          fn=lazy_fns.trace(MockAverageFn)(),
          expected={'a': [2]},
      ),
      dict(
          testcase_name='agg_with_fn_instance',
          inputs={'a': [0, 1, 2], 'b': [1, 2, 3]},
          input_keys='b',
          output_keys='a',
          fn=MockAverageFn(),
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
    t1 = transform.TreeTransform().aggregate(
        fn, input_keys=input_keys, output_keys=output_keys
    )
    t2 = transform.TreeTransform().aggregate(
        fn, input_keys=input_keys, output_keys=output_keys
    )
    agg1, agg2 = t1.make(), t2.make()
    state1 = agg1.update_state(agg1.create_state(), inputs)
    state2 = agg2.update_state(agg2.create_state(), inputs)
    # LazyFn of the fn enables across workers merge since these are consistent
    # after reinstantion of the actual function instance.
    merged_t1 = agg1.merge_states([state1, state2])
    merged_t2 = agg2.merge_states([state1, state2])
    self.assertEqual(merged_t1, merged_t2)
    self.assertEqual(expected, agg1.get_result(state1))
    self.assertEqual(expected, agg2.get_result(state2))

  def test_aggregate_with_slices_single(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 1, 1]}
    agg = (
        transform.TreeTransform()
        .aggregate(
            MockAverageFn(),
            input_keys='a',
            output_keys='avg_a',
        )
        .add_slice('b')
    )
    expected = {
        'avg_a': [4.0 / 3],
        MetricKey('avg_a', SliceKey(('b',), (1,))): [4.0 / 3],
    }
    actual = agg.make()(inputs)
    self.assertEqual(expected, actual)

  def test_aggregate_result_size(self):
    inputs = [1, 2, 1]
    agg = transform.TreeTransform().aggregate(MockAverageFn(return_tuple=True))
    it = agg.make().iterate([inputs])
    _ = mit.last(it)
    self.assertLen(it._iterators[0].agg_result, 1)
    self.assertEqual({'': ([4.0 / 3], 0.0)}, it.agg_result)

  def test_aggregate_with_slices_multiple_aggs(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 9, 5]}
    agg = (
        transform.TreeTransform()
        .aggregate(
            MockAverageFn(),
            input_keys='a',
            output_keys='avg_a',
            disable_slicing=True,
        )
        .add_aggregate(
            MockAverageFn(return_dict_key='avg_result'),
            input_keys='b',
            output_keys=dict(avg_b='avg_result'),
        )
        .add_slice('a')
        .add_slice('b', replace_mask_false_with=0)
    )
    expected = {
        'avg_a': [4.0 / 3],
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a',), (1,))): [3.0],
        MetricKey('avg_b', SliceKey(('a',), (2,))): [9.0],
        MetricKey('avg_b', SliceKey(('b',), (1,))): [1 / 3],
        MetricKey('avg_b', SliceKey(('b',), (9,))): [9 / 3],
        MetricKey('avg_b', SliceKey(('b',), (5,))): [5 / 3],
    }
    actual = agg.make()(inputs)
    self.assertEqual(expected, actual)

  @parameterized.named_parameters([
      dict(
          testcase_name='default_metric_name_with_slice',
          add_slice=True,
          expected={'': [1], MetricKey('', SliceKey(('b',), (1,))): [1]},
      ),
      dict(
          testcase_name='default_metric_name_no_slice',
          add_slice=False,
          expected={'': [1]},
      ),
  ])
  def test_aggregate_with_slices_default_metric_name(self, add_slice, expected):
    p = transform.TreeTransform().aggregate(
        MockAverageFn(),
        input_keys='a',
    )
    if add_slice:
      p = p.add_slice('b')
    actual = p.make()({'a': [1, 1, 1], 'b': [1, 1, 1]})
    self.assertEqual(expected, actual)

  def test_aggregate_with_slice_within_values(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 9, 5]}
    agg = (
        transform.TreeTransform()
        .aggregate(MockAverageFn(), input_keys='b', output_keys='avg_b')
        .add_slice(dict(a=1))
        .add_slice(dict(b=(1, 9)))
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a',), (1,))): [3.0],
        MetricKey('avg_b', SliceKey(('b',), (1,))): [1.0],
        MetricKey('avg_b', SliceKey(('b',), (9,))): [9.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_with_slicing_on_dict(self):
    inputs = {'a': np.array([1, 2, 1]), 'b': np.array([1, 9, 5])}
    agg = (
        transform.TreeTransform().agg(
            MockAverageFn(input_key='b'), output_keys='avg_b'
        )
        # x.item() is needed to convert a non-hashable np.array to a hashable
        # one. While np.array scalar is hashable, it is not hashable in
        # Jax numpy array.
        .add_slice('a', slice_fn=lambda x: (x.item(),))
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a',), (1,))): [3.0],
        MetricKey('avg_b', SliceKey(('a',), (2,))): [9.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_slice_fn_fanout_with_multiple_inputs(self):
    # Here is an example of using slice_fn to fanout the input as slices.
    # Here, each example generates two slices a and then b by slice_fn, the new
    # slice is named as 'a_or_b'. The following explains the computations:
    # Inputs after applying slice_fn.
    # a    b   a_or_b
    # 1    1   [1, 1]
    # 2    9   [2, 9]
    # 1    5   [1, 5]
    # This fans out to:
    # a    b   a_or_b
    # 1    1     1
    # 1    1     1
    # 2    9     2
    # 2    9     9
    # 1    5     1
    # 1    5     5
    # After group by:
    # a_or_b
    #   1       [{a: 1, b: 1}, {a: 1, b: 1} (duplicate), {a: 1, b: 5}]
    #   2       [{a: 2, b: 9}]
    #   5       [{a: 1, b: 5}]
    #   9       [{a: 2, b: 9}]
    # We are calculating the average of b for each a_or_b value. So we get:
    # a_or_b  avg_b
    # 1        (1 + 5) / 2 = 3
    # 2        9
    # 5        3
    # 9        9
    inputs = {'a': np.array([1, 2, 1]), 'b': np.array([1, 9, 5])}
    agg = (
        transform.TreeTransform().agg(
            MockAverageFn(input_key='b'), output_keys='avg_b'
        )
        # x.item() is needed to convert a non-hashable np.array to a hashable
        # one. While np.array scalar is hashable, it is not hashable in
        # Jax numpy array.
        .add_slice(
            ('a', 'b'),
            slice_name='a_or_b',
            slice_fn=lambda x, y: (x.item(), y.item()),
        )
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a_or_b',), (1,))): [3.0],
        MetricKey('avg_b', SliceKey(('a_or_b',), (2,))): [9.0],
        MetricKey('avg_b', SliceKey(('a_or_b',), (5,))): [5.0],
        MetricKey('avg_b', SliceKey(('a_or_b',), (9,))): [9.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_slice_fn_with_crosses(self):
    inputs = {'a': np.array([1, 2, 1]), 'b': np.array([1, 9, 5])}
    agg = (
        transform.TreeTransform().agg(
            MockAverageFn(input_key='b'), output_keys='avg_b'
        )
        # x.item() is needed to convert a non-hashable np.array to a hashable
        # one. While np.array scalar is hashable, it is not hashable in
        # Jax numpy array.
        .add_slice(
            ('a', 'b'),
            slice_name=('a_', 'b_'),
            slice_fn=lambda x, y: ((str(x), str(y)),),
        )
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a_', 'b_'), ('1', '1'))): [1.0],
        MetricKey('avg_b', SliceKey(('a_', 'b_'), ('2', '9'))): [9.0],
        MetricKey('avg_b', SliceKey(('a_', 'b_'), ('1', '5'))): [5.0],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_with_slice_crosses(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 9, 5]}
    agg = (
        transform.TreeTransform()
        .aggregate(MockAverageFn(), input_keys='b', output_keys='avg_b')
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
        transform.TreeTransform()
        .agg(MockAverageFn(), input_keys='b', output_keys='avg_b')
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
        transform.TreeTransform()
        .aggregate(MockAverageFn(), input_keys='b', output_keys='avg_b')
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
        transform.TreeTransform()
        .aggregate(MockAverageFn(), input_keys='b', output_keys='avg_b')
        .add_slice('a', slice_fn=foo, slice_name=('pos', 'a'))
    )
    with self.assertRaises(ValueError):
      agg.make()(inputs)

  def test_input_iterator_transform(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform()
        .data_source(test_utils.NoLenIter(input_iterator))
        .apply(sum)
    )
    actual_fn = t.make()
    iterator = transform.iterate_with_returned(actual_fn)
    self.assertEqual([6, 9, None], list(iterator))
    self.assertEqual([6, 9], list(actual_fn.iterate(input_iterator)))

  def test_input_iterator_aggregate(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform()
        .data_source(lazy_fns.trace(test_utils.NoLenIter)(input_iterator))
        .apply(iter_utils.iterate_fn(lambda x: x + 1))
        .apply(iter_utils.iterate_fn(lambda x: x + 10))
        .agg(MockAverageFn())
    )
    actual_fn = t.make()
    self.assertEqual({'': [13.5]}, actual_fn())
    actual = actual_fn(input_iterator=test_utils.NoLenIter(input_iterator))
    self.assertEqual({'': [13.5]}, actual)
    it = actual_fn.iterate()
    # Aggregation state is accumulated when iterating.
    self.assertIsNotNone(it.agg_state)
    self.assertDictEqual({'': [np.nan]}, it.agg_result)
    _ = next(it)
    assert it.agg_state is not None
    # First batch is [1, 2, 3] + 11 = [12, 13, 14], state is (sum=39, count=3).
    self.assertEqual((39, 3), next(iter(it.agg_state.values())))
    self.assertDictAlmostEqual({'': [3.0 + 10]}, it.agg_result)
    # Exhausting the iterator is necessary to get the aggregate result.
    mit.last(it)
    self.assertEqual((81, 6), next(iter(it.agg_state.values())))
    self.assertDictAlmostEqual({'': [13.5]}, it.agg_result)

    mit.last(result := actual_fn.iterate(test_utils.NoLenIter(input_iterator)))
    self.assertDictAlmostEqual({'': [13.5]}, result.agg_result)

  def test_multiple_aggregates_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'more than one aggregations'):
      _ = (
          transform.TreeTransform()
          .apply(iter_utils.iterate_fn(lambda x: x + 1))
          .agg(MockAverageFn())
          .agg(MockAverageFn())
      )

  def test_input_iterator_aggregate_incorrect_states_count_raises_error(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform()
        .data_source(lazy_fns.trace(test_utils.NoLenIter)(input_iterator))
        .apply(iter_utils.iterate_fn(lambda x: x + 1))
        .apply(iter_utils.iterate_fn(lambda x: x + 10))
        .agg(MockAverageFn())
    )
    actual_fn = t.make()
    with self.assertRaises(ValueError):
      # Exhausting the iterator is necessary to get the aggregate result.
      mit.last(state := actual_fn.iterate())
      assert state.agg_state is not None
      actual_fn.merge_states([state.agg_state], strict_states_cnt=2)

  @parameterized.named_parameters([
      dict(
          testcase_name='split_agg',
          split_agg=True,
      ),
      dict(
          testcase_name='normal',
          split_agg=False,
      ),
  ])
  def test_flatten_transform(self, split_agg):
    t = (
        transform.TreeTransform()
        # consecutive datasource, asign, and apply form one tranform.
        .data_source(lazy_fns.trace(test_utils.NoLenIter)(range(6)))
        .apply(lambda x: x + 1, output_keys='a')
        .assign('b', fn=lambda x: x + 10, input_keys='a')
        .batch(2)
        .agg(MockAverageFn(), input_keys='b', output_keys='avg_b')
        .add_agg(fn=MockAverageFn(), input_keys='a', output_keys='avg_a')
    )
    it = t.make().iterate()
    results = list(it)
    agg_result = it.agg_result

    transforms = t.flatten_transform(split_agg=split_agg)
    self.assertTrue(all(t is not None for t in transforms))
    if split_agg:
      self.assertLen(transforms, 2)
      self.assertEmpty(transforms[0].agg_fns)
      self.assertNotEmpty(transforms[1].agg_fns)
      it = transforms[0].make().iterate()
      it = transforms[1].make().iterate(it)
      actual, actual_agg = list(it), it.agg_result
      self.assertEqual(actual, results)
      self.assertEqual(actual_agg, agg_result)
    else:
      self.assertLen(transforms, 1)
      self.assertNotEmpty(transforms[0].agg_fns)
      self.assertIs(transforms[0], t)

  def test_flatten_transform_with_empty_transform(self):
    self.assertEmpty(transform.TreeTransform().flatten_transform())
    self.assertEmpty(transform.TreeTransform().agg().flatten_transform())
    self.assertEmpty(transform.TreeTransform().apply().flatten_transform())

  @parameterized.named_parameters([
      dict(
          testcase_name='fused',
          names=('', ''),
          expected_transforms=[''],
      ),
      dict(
          testcase_name='chain_unfused',
          names=('a', 'b'),
          expected_transforms=['a', 'b'],
      ),
  ])
  def test_chain(self, names, expected_transforms):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    read = transform.TreeTransform(name=names[0]).data_source(
        lazy_fns.trace(test_utils.NoLenIter)(input_iterator)
    )
    process = (
        transform.TreeTransform(name=names[1])
        .apply(iter_utils.iterate_fn(lambda x: x + 1))
        .apply(iter_utils.iterate_fn(lambda x: x + 10))
        .agg(MockAverageFn())
    )
    t = read.chain(process)
    actual_fn = t.make()
    it = t.make().iterate()
    self.assertLen(it._iterators, len(set(names)))
    self.assertEqual([[12, 13, 14], [13, 14, 15]], list(it))
    self.assertEqual({'': [13.5]}, actual_fn())
    self.assertEqual({'': [13.5]}, actual_fn(input_iterator=input_iterator))
    self.assertLen(t.flatten_transform(), len(expected_transforms))
    self.assertEqual(expected_transforms, list(t.named_transforms()))

  @parameterized.named_parameters([
      dict(
          testcase_name='no_threads',
          num_threads=0,
      ),
      dict(
          testcase_name='with_one_thread',
          num_threads=1,
      ),
      dict(
          testcase_name='with_threads',
          num_threads=8,
      ),
  ])
  def test_transform_with_multiple_threads(self, num_threads):
    inputs = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    p = (
        transform.TreeTransform()
        .data_source(test_utils.NoLenIter(inputs))
        .apply(lambda x: x)
        .agg(MockAverageFn())
    )
    actual = p.make(num_threads=num_threads)()
    self.assertEqual({'': [3.0]}, actual)

  def test_iterator_queue_with_transform(self):
    inputs = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    p = (
        transform.TreeTransform()
        .data_source(test_utils.NoLenIter(inputs))
        .agg(MockAverageFn())
    )
    iterator = iter_utils.IteratorQueue(2)
    with futures.ThreadPoolExecutor() as thread_pool:
      thread_pool.submit(iterator.enqueue_from_iterator, p.make())
      result = iterator.get_batch(min_batch_size=len(inputs))
    self.assertEqual(inputs, result)
    self.assertEqual({'': [3.0]}, iterator.returned[0].agg_result)

  def test_transform_maybe_stop(self):
    datasource = test_utils.range_with_sleep(256, 0.3)
    p_ds = transform.TreeTransform(num_threads=1).data_source(datasource)
    p_apply = transform.TreeTransform(name='apply', num_threads=1).apply(
        lambda x: x
    )
    p = p_ds.interleave(p_apply)

    it = p.make().iterate()
    for x in it:
      if x == 2:
        it.maybe_stop()
    self.assertLen(it._iterators, 2)
    threads = _get_all_threads(it._iterators)
    self.assertNotEmpty(threads)
    self.assertTrue(all(not t.is_alive() for t in threads))

  def test_transform_len(self):
    inputs = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    p = (
        transform.TreeTransform()
        .data_source(test_utils.NoLenIter(inputs))
    )
    self.assertLen(p.make().iterate(data_source_size=len(inputs)), len(inputs))


if __name__ == '__main__':
  absltest.main()
