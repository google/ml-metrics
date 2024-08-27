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

from collections.abc import Callable, Iterable
import dataclasses
import functools
import itertools
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.aggregates import base as aggretates
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import transform
from ml_metrics._src.chainables import tree
from ml_metrics._src.chainables import tree_fns
from ml_metrics._src.utils import iter_utils
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


def infinite_generator(n):
  yield from range(n)
  raise ValueError(
      'This is to test the runner does not iterate and copy everything into'
      ' memory, it should never be exhausted.',
      'This is to test the runner does not iterate and copy everything into'
      ' memory, it should never be exhausted.',
  )


class MockGenerator(Iterable):

  def __init__(self, iterable):
    self._iteratable = iterable

  def __len__(self):
    raise NotImplementedError()

  def __iter__(self):
    return iter(self._iteratable)


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


# TODO: b/318463291 - Improves test coverage.
class MockAverageFn:

  def __init__(self, batch_output=True, return_tuple=False, input_key=None):
    self.batch_output = batch_output
    self.return_tuple = return_tuple
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
    result = state[0] / state[1]
    result = [result] if self.batch_output else result
    return (result, 0) if self.return_tuple else result


@dataclasses.dataclass
class MockPrecisionRecall:
  matcher: Callable[[Any, Any], Any] | None = None
  pred_tp: int = 0
  gp_tp: int = 0
  pred_p: int = 0
  gt_p: int = 0

  def make(self):
    return MockPrecisionRecall(self.matcher)

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


class TransformTest(parameterized.TestCase):

  def assert_nested_sequence_equal(self, a, b):
    self.assertEqual(type(a), type(b))
    if isinstance(a, dict):
      for (k_a, v_a), (k_b, v_b) in zip(
          sorted(a.items()), sorted(b.items()), strict=True
      ):
        self.assertEqual(k_a, k_b)
        self.assert_nested_sequence_equal(v_a, v_b)
    elif isinstance(a, str):
      self.assertEqual(a, b)
    elif isinstance(a, Iterable):
      for a_elem, b_elem in zip(a, b, strict=True):
        self.assert_nested_sequence_equal(a_elem, b_elem)
    else:
      self.assertEqual(a, b)

  def test_transform_iterator_pipe_with_data_source(self):
    t = (
        transform.TreeTransform.new()
        .data_source(MockGenerator(range(3)))
        .apply(fn=lambda x: x + 1)
    )
    pipe = t.make().iterator_pipe(timeout=3)
    self.assertIsNone(pipe.input_queue)
    self.assertIsNotNone(pipe.output_queue)
    actual = list(pipe.output_queue.dequeue_as_iterator())
    self.assertEqual([1, 2, 3], actual)

  def test_transform_iterator_pipe_normal(self):
    t = transform.TreeTransform.new().apply(fn=lambda x: x + 1)
    pipe = t.make().iterator_pipe(timeout=3)
    self.assertIsNotNone(pipe.input_queue)
    self.assertIsNotNone(pipe.output_queue)
    pipe.input_queue.enqueue_from_iterator(range(3))
    actual = list(pipe.output_queue.dequeue_as_iterator())
    self.assertEqual([1, 2, 3], actual)

  def test_transform_unique_ids_with_each_op(self):
    seen_ids = set()
    t = transform.TreeTransform.new()
    seen_ids.add(t.id)
    t = dataclasses.replace(t, name='foo')
    seen_ids.add(t.id)
    t = t.apply(output_keys='a', fn=lambda x: x + 1)
    seen_ids.add(t.id)
    t = t.assign('b', fn=lambda x: x + 1)
    seen_ids.add(t.id)
    t = dataclasses.replace(t, name='')
    t = t.aggregate(output_keys='c', fn=lambda x: x + 1)
    seen_ids.add(t.id)
    t = t.add_aggregate(output_keys='d', fn=MockAverageFn()).add_slice('a')
    seen_ids.add(t.id)
    # Each new transform should create a unique id.
    self.assertLen(seen_ids, 6)

  def test_transform_call(self):
    t = (
        transform.TreeTransform.new()
        .data_source(MockGenerator(range(3)))
        .apply(fn=lambda x: x + 1)
    )
    self.assertEqual(t.make()(), [1, 2, 3])
    self.assertEqual([2, 3, 4], t.make()(input_iterator=range(1, 4)))
    self.assertEqual(2, t.make()(1))

  def test_cached_make(self):
    t = (
        transform.TreeTransform.new(use_cache=True)
        .data_source(MockGenerator(range(3)))
        .apply(fn=lambda x: x + 1)
        .aggregate(fn=lazy_fns.trace(MockAverageFn)())
        .apply(fn=lambda x: x + 1)
    )
    pickled_t = lazy_fns.pickler.dumps(lazy_fns.trace_object(t).make())
    self.assertIsInstance(
        lazy_fns.maybe_make(pickled_t), transform.CombinedTreeFn
    )
    lazy_fns.maybe_make(pickled_t)
    with mock.patch.object(
        transform, '_transform_make', autospec=True
    ) as mock_make_transform:
      lazy_fns.maybe_make(pickled_t)
      mock_make_transform.assert_not_called()

  def test_transform_named_transforms_default(self):
    t1 = (
        transform.TreeTransform.new()
        .data_source(MockGenerator(range(3)))
        .apply(fn=lambda x: x + 1)
    )
    t2 = transform.TreeTransform.new(name='A').aggregate(
        fn=lazy_fns.trace(MockAverageFn)()
    )
    t3 = (
        transform.TreeTransform.new(name='B')
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
    )
    t = t1.chain(t2).chain(t3)
    self.assertSameElements(t.named_transforms().keys(), (t1.id, 'A', 'B'))
    # The first group of nodes are identical.
    self.assertEqual(t.named_transforms()[t1.id], t1)
    self.assertEqual(t.named_transforms()[t1.id].make()(), t1.make()())
    inputs = [1, 2, 3]
    self.assertEqual(
        t.named_transforms()['A'].make()(inputs), t2.make()(inputs)
    )
    inputs = [1, 2, 3]
    self.assertEqual(
        t.named_transforms()['B'].make()(inputs), t3.make()(inputs)
    )

  def test_transform_named_transforms_with_duplicate_names(self):
    t1 = transform.TreeTransform.new(name='A').data_source(
        MockGenerator(range(3))
    )
    t2 = transform.TreeTransform.new(name='B').aggregate(
        fn=lazy_fns.trace(MockAverageFn)()
    )
    t3 = transform.TreeTransform.new(name='A').apply(
        fn=lazy_fns.iterate_fn(lambda x: x + 1)
    )
    t = t1.chain(t2).chain(t3)
    with self.assertRaisesRegex(ValueError, 'Duplicate transform'):
      t.named_transforms()

  def test_transform_equal(self):
    t = transform.TreeTransform.new()
    pickled_t = lazy_fns.pickler.dumps(lazy_fns.trace_object(t))
    self.assertEqual(
        lazy_fns.maybe_make(pickled_t),
        lazy_fns.maybe_make(pickled_t),
    )

  def test_transform_iterate_is_not_copy(self):
    t = (
        transform.TreeTransform.new()
        .data_source(infinite_generator(2))
        .apply(fn=lambda x: x + 1, output_keys='a')
        .assign('b', fn=lambda x: x + 1, input_keys='a')
    )
    actual = list(itertools.islice(t.make().iterate(), 2))
    self.assertEqual(actual, [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}])

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
    t = transform.TreeTransform.new().apply(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
        fn_batch_size=fn_batch_size,
        batch_size=batch_size,
    )
    actual = t.make()(input_iterator=inputs)
    self.assert_nested_sequence_equal(expected, actual)

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
      dict(
          testcase_name='without_keys',
          inputs=map(list, mit.batched(range(5), 3)),
          fn_batch_size=2,
          batch_size=3,
          fn=BatchedCall(batch_size=2),
          expected=[np.array([1, 2, 3]), np.array([4, 5])],
      ),
      dict(
          testcase_name='rebatch_only',
          inputs=map(list, mit.batched(range(5), 2)),
          batch_size=3,
          expected=[[0, 1, 2], [3, 4]],
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
          testcase_name='input_keys_only',
          inputs=[{'a': [0, 1, 2], 'b': [1]}, {'a': [4, 5], 'b': [5]}],
          fn_batch_size=2,
          batch_size=3,
          input_keys='a',
          fn=BatchedCall(batch_size=2),
          expected=[np.array([1, 2, 3]), np.array([5, 6])],
      ),
  ])
  def test_assign_transform_rebatched(
      self,
      inputs,
      expected,
      fn=None,
      fn_batch_size=0,
      batch_size=0,
      input_keys=tree.Key.SELF,
      output_keys=tree.Key.SELF,
  ):
    t = transform.TreeTransform.new().assign(
        output_keys=output_keys,
        fn=fn,
        input_keys=input_keys,
        fn_batch_size=fn_batch_size,
        batch_size=batch_size,
    )
    actual = t.make()(input_iterator=inputs)
    self.assert_nested_sequence_equal(expected, actual)

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
    t = transform.TreeTransform.new().select(
        input_keys,
        output_keys=output_keys,
    )
    self.assertEqual(expected, t.make()(inputs))

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
          fn=MockAverageFn(),
      ).add_aggregate(fn=MockAverageFn(), output_keys='a')

    with self.assertRaises(ValueError):
      transform.TreeTransform.new().aggregate(
          fn=MockAverageFn(),
          output_keys='a',
      ).add_aggregate(fn=MockAverageFn())

  def test_aggregate_duplicate_keys(self):
    with self.assertRaises(ValueError):
      transform.TreeTransform.new().aggregate(
          fn=MockAverageFn(),
          output_keys='a',
      ).add_aggregate(fn=MockAverageFn(), output_keys=('a', 'b'))

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
            fn=MockAverageFn(),
            output_keys='avg_b',
        )
        .add_slice('a')
        .add_slice('b', replace_mask_false_with=0)
    )
    expected = {
        'avg_b': [5.0],
        MetricKey('avg_b', SliceKey(('a',), (1,))): [3.0],
        MetricKey('avg_b', SliceKey(('a',), (2,))): [9.0],
        MetricKey('avg_b', SliceKey(('b',), (1,))): [1 / 3],
        MetricKey('avg_b', SliceKey(('b',), (9,))): [9 / 3],
        MetricKey('avg_b', SliceKey(('b',), (5,))): [5 / 3],
    }
    self.assertEqual(expected, agg.make()(inputs))

  def test_aggregate_with_slice_within_values(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 9, 5]}
    agg = (
        transform.TreeTransform.new()
        .aggregate(
            input_keys='b',
            fn=MockAverageFn(),
            output_keys='avg_b',
        )
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
        transform.TreeTransform.new().aggregate(
            fn=MockAverageFn(input_key='b'),
            output_keys='avg_b',
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

  def test_aggregate_with_slice_crosses(self):
    inputs = {'a': [1, 2, 1], 'b': [1, 9, 5]}
    agg = (
        transform.TreeTransform.new()
        .aggregate(
            input_keys='b',
            fn=MockAverageFn(),
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
            fn=MockAverageFn(),
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
            fn=MockAverageFn(),
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
            fn=MockAverageFn(),
            output_keys='avg_b',
        )
        .add_slice('a', slice_fn=foo, slice_name=('pos', 'a'))
    )
    with self.assertRaises(ValueError):
      agg.make()(inputs)

  @parameterized.named_parameters([
      dict(
          testcase_name='single_mask_on_pred',
          slicer_input='attr_pred',
          slice_mask_fn=functools.partial(single_slicer, within=('a', 'b')),
          slice_name='pred_class',
          expected={
              'precision': 0.75,
              'recall': 0.6,
              MetricKey('precision', SliceKey(('pred_class',), ('a',))): 0.5,
              MetricKey('precision', SliceKey(('pred_class',), ('b',))): 1.0,
              MetricKey('recall', SliceKey(('pred_class',), ('a',))): 0.6,
              MetricKey('recall', SliceKey(('pred_class',), ('b',))): 0.6,
          },
      ),
      dict(
          testcase_name='single_mask_on_label',
          slicer_input='attr_label',
          slice_mask_fn=functools.partial(
              single_slicer, for_pred=False, within=('e', 'g')
          ),
          slice_name='label_class',
          expected={
              'precision': 0.75,
              'recall': 0.6,
              MetricKey('precision', SliceKey(('label_class',), ('e',))): 0.75,
              MetricKey('precision', SliceKey(('label_class',), ('g',))): 0.75,
              MetricKey('recall', SliceKey(('label_class',), ('e',))): 1.0,
              MetricKey('recall', SliceKey(('label_class',), ('g',))): 0.5,
          },
      ),
      dict(
          testcase_name='dual_mask_on_pred_and_label',
          slicer_input=('pred', 'label'),
          slice_mask_fn=functools.partial(multi_slicer, within=(0, 1)),
          slice_name='classes',
          expected={
              'precision': 0.75,
              'recall': 0.6,
              MetricKey('precision', SliceKey(('classes',), (0,))): 2 / 3,
              MetricKey('precision', SliceKey(('classes',), (1,))): 1 / 1,
              MetricKey('recall', SliceKey(('classes',), (0,))): 1 / 1,
              MetricKey('recall', SliceKey(('classes',), (1,))): 2 / 3,
          },
      ),
  ])
  def test_intra_example_slicing(
      self, slicer_input, slice_mask_fn, slice_name, expected
  ):
    inputs = [{
        'pred': [[0, 0], [0, 1]],
        'label': [[0, 1, 2], [1, 1]],
        # This attributes only align with prediction.
        'attr_pred': [['a', 'b'], ['a', 'b']],
        # This attributes only align with label attributes.
        'attr_label': [['e', 'f', 'g'], ['f', 'g']],
    }]

    # Matcher results:
    # 'pred_tp': [[True, True], [False, True]]
    # 'label_tp:  [[True, False, False], [True, True]]
    # Aggregate results:
    # Overall precision:  3 / 4 = 0.75
    # Overall recall:    3 / 5 = 0.6
    # slice on 'attr_pred' 'a':
    #   precision: 1 / 2 = 0.5
    #   recall:    0.6 (same as overall)
    # slice on 'attr_pred' 'b':
    #   precision: 2 / 2 = 1.0
    #   recall:   0.6 (same as overall)
    # slice on 'attr_label' 'e':
    #   precision: 0.75 (same as overall)
    #   recall:    1 / 1 = 1.0
    # slice on 'attr_label' 'g':
    #   precision: 0.75 (same as overall)
    #   recall:   1 / 2 = 0.5
    # slice on 'classes' 0:
    #   precision: 1 / 1
    #   recall:   1 / 1
    # slice on 'classes' 1:
    #   precision: 1 / 1
    #   recall:   2 / 3

    def matcher(pred, label):
      # We need two tp table to avoid deduping for different derived metrics.
      # e.g., precision only uses pred_tp, and recall only uses label_tp.
      pred_tp = [elem in label for elem in pred]
      label_tp = [elem in pred for elem in label]
      return pred_tp, label_tp

    # Without slicing, the aggregate function can use the matcher directly.
    agg_overall = (
        transform.TreeTransform.new()
        .data_source(inputs)
        .aggregate(
            output_keys=('precision', 'recall'),
            fn=aggretates.MergeableMetricAggFn(
                MockPrecisionRecall(lazy_fns.iterate_fn(matcher))
            ),
            # Provide matched result directly to bypass internal matcher.
            input_keys=('pred', 'label'),
        )
    ).make()()
    self.assertEqual(
        {
            'precision': 0.75,
            'recall': 0.6,
        },
        agg_overall,
    )

    # With slicing, the matcher logic needs to be separated from the aggregate
    # function, so that slicing can be applied to the matched result.
    agg = (
        transform.TreeTransform.new()
        .data_source(inputs)
        .assign(
            ('pred_matched', 'label_matched'),
            fn=lazy_fns.iterate_fn(matcher),
            input_keys=('pred', 'label'),
        )
        .aggregate(
            output_keys=('precision', 'recall'),
            fn=aggretates.MergeableMetricAggFn(MockPrecisionRecall()),
            # Provide matched result directly to bypass internal matcher.
            input_keys=dict(
                matched_pred='pred_matched', matched_label='label_matched'
            ),
        )
        .add_slice(
            slicer_input,
            slice_mask_fn=slice_mask_fn,
            slice_name=slice_name,
        )
    )
    agg_result = agg.make()()
    self.assertEqual(expected, agg_result)

  def test_input_iterator_transform(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform.new()
        .data_source(iterator=MockGenerator(input_iterator))
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
        .data_source(iterator=lazy_fns.trace(MockGenerator)(input_iterator))
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
        .aggregate(fn=MockAverageFn())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 10))
        .aggregate(fn=MockAverageFn())
    )
    actual_fn: transform.CombinedTreeFn = t.make()
    self.assertEqual([13.5], actual_fn())
    self.assertEqual(
        [13.5], actual_fn(input_iterator=MockGenerator(input_iterator))
    )

    result = transform.get_generator_returned(
        actual_fn.iterate(with_agg_result=True)
    )
    self.assertEqual([13.5], result.agg_result)

    result = transform.get_generator_returned(
        actual_fn.iterate(
            input_iterator=MockGenerator(input_iterator), with_agg_result=True
        )
    )
    self.assertEqual([13.5], result.agg_result)

  def test_input_iterator_aggregate_incorrect_states_count_raises_error(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform.new()
        .data_source(iterator=lazy_fns.trace(MockGenerator)(input_iterator))
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
        .aggregate(fn=MockAverageFn())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 10))
        .aggregate(fn=MockAverageFn())
    )
    actual_fn: transform.CombinedTreeFn = t.make()
    with self.assertRaises(ValueError):
      state = transform.get_generator_returned(
          actual_fn.iterate(with_agg_state=True)
      )
      assert state is not None
      actual_fn.merge_states([state.agg_state], strict_states_cnt=2)

  def test_flatten_transform(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    t = (
        transform.TreeTransform.new()
        # consecutive datasource, asign, and apply form one tranform.
        .data_source(iterator=lazy_fns.trace(MockGenerator)(input_iterator))
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1), output_keys='a')
        .assign('b', fn=lazy_fns.iterate_fn(lambda x: x + 1), input_keys='a')
        # aggregate is a new tranform.
        .aggregate(fn=MockAverageFn(), input_keys='b')
        # asign or apply after aggregate is a new tranform.
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 10))
        # aggregate is a new tranform.
        .aggregate(fn=MockAverageFn())
    )

    transforms = t.flatten_transform()
    self.assertTrue(all(t is not None for t in transforms))
    self.assertLen(transforms, 4)
    self.assertSequenceEqual(transforms, t.flatten_transform())
    other_transforms = t.flatten_transform(remove_input_transform=True)
    self.assertTrue(all(t is not None for t in other_transforms))
    self.assertEqual(transforms[0], other_transforms[0])
    self.assertTrue(
        all(
            x != y
            for x, y in zip(transforms[1:], other_transforms[1:], strict=True)
        )
    )

  def test_chain(self):
    input_iterator = [[1, 2, 3], [2, 3, 4]]
    read = transform.TreeTransform.new().data_source(
        iterator=lazy_fns.trace(MockGenerator)(input_iterator)
    )
    process = (
        transform.TreeTransform.new()
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 1))
        .aggregate(fn=MockAverageFn())
        .apply(fn=lazy_fns.iterate_fn(lambda x: x + 10))
        .aggregate(fn=MockAverageFn())
    )
    t = read.chain(process)
    actual_fn = t.make()
    self.assertEqual([13.5], actual_fn())
    self.assertEqual([13.5], actual_fn(input_iterator=input_iterator))

  def test_prefetched_iterator(self):
    inputs = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    p = (
        transform.TreeTransform.new()
        .data_source(iterator=MockGenerator(inputs))
        .aggregate(fn=MockAverageFn())
    )
    iterator = iter_utils.PrefetchedIterator(
        p.make().iterate(with_agg_result=True), prefetch_size=2
    )
    iterator.prefetch()
    self.assertEqual(2, iterator.cnt)
    self.assertEqual(inputs[:2], iterator.flush_prefetched())
    self.assertEqual([inputs[-1]], list(iterator))
    self.assertEqual([3.0], iterator.returned.agg_result)


if __name__ == '__main__':
  absltest.main()
