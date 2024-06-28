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
import types
from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.chainables import tree
import numpy as np

Key = tree.Key
TreeMapView = tree.TreeMapView


class PathTest(absltest.TestCase):

  def test_path_as_tuple(self):
    self.assertEqual((), Key.new())
    self.assertEqual(('a',), Key(('a',)))
    self.assertEqual(('a', 3), Key(('a', 3)))
    self.assertEqual(('a', 3, Key.Index(1)), Key(('a', 3, Key.Index(1))))

  def test_path_by_new(self):
    self.assertEqual((), Key.new())
    self.assertEqual(('a',), Key.new('a'))
    self.assertEqual(('a', 3), Key.new('a', 3))
    self.assertEqual(('a', 3, Key.Index(1)), Key.new('a', 3, Key.Index(1)))

  def test_path_by_suffix(self):
    self.assertEqual(('a', 'b', 'c'), Key().a.b.c)
    self.assertEqual((0, 'a', 'b'), Key().at(0).a.b)
    self.assertEqual((Key.Index(1), 'a'), Key().at(Key.Index(1)).a)
    self.assertEqual(('a', 0, Key.Index(1)), Key().a.at(0).at(Key.Index(1)))

  def test_index_not_int(self):
    self.assertIsNot(Key.Index(1), 1)
    self.assertNotIsInstance(1, tree.Index)

  def test_index_repr(self):
    self.assertEqual('Index(1)', repr(Key.Index(1)))

  def test_path_repr(self):
    self.assertEqual("Path(('a', 'b'))", repr(Key(('a', 'b'))))

  def test_reserved_repr(self):
    self.assertEqual("Reserved('SELF')", repr(Key.SELF))
    self.assertEqual("Reserved('SKIP')", repr(Key.SKIP))

  def test_reserved_iter(self):
    self.assertEqual([Key.SELF], list(Key.SELF))
    self.assertEqual([Key.SKIP], list(Key.SKIP))


class TreeMapViewTest(absltest.TestCase):

  def test_as_view(self):
    data = {'a': [1, 2], 'b': [3, 4]}
    view = TreeMapView.as_view(data)
    self.assertEqual(([1, 2], 3), view['a', Key.Literal(3)])
    self.assertEqual(([1, 2], [3, 4]), view['a', 'b'])
    self.assertIsNot(data, view)
    self.assertEqual(view, TreeMapView.as_view(view))

  def test_get_by_reserved_key(self):
    data = {'a': [1, 2], 'b': [3, 4]}
    view = TreeMapView.as_view(data)
    self.assertEqual(data, view[Key.SELF])
    self.assertEqual([1, 2], view[Key.new('a', Key.SELF)])

  def test_get_by_skip_raise_error(self):
    with self.assertRaises(KeyError):
      TreeMapView({'a': 1})[Key.SKIP, 'a']  # pylint: disable=expression-not-assigned

  def test_key_path(self):
    view = TreeMapView({'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}]})
    self.assertEqual([1, 2], view[Key.new().a.a1])
    self.assertEqual(['c', 'd'], view[Key.new('b', Key.Index(0), 0)])
    self.assertEqual(view.data, view[Key.SELF])
    self.assertEqual(view.data, view[Key()])
    self.assertEqual((), view[()])

  def test_incorrect_keys_raises_error(self):
    # pylint: disable=expression-not-assigned
    with self.assertRaises(ValueError):
      TreeMapView([1, 2, 3])[0]
    with self.assertRaises(IndexError):
      TreeMapView([1, 2, 3])[Key.Index(9)]
    with self.assertRaises(KeyError):
      TreeMapView({'a': 1})['b']
    with self.assertRaises(TypeError):
      TreeMapView({'a': 1})[set('a')]
    # pylint: enable=expression-not-assigned

  def test_iter(self):
    data = {'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}], 'c': {}, 'e': []}
    view = TreeMapView(data)
    self.assertEqual(
        [
            Key.new('a', 'a1', Key.Index(0)),
            Key.new('a', 'a1', Key.Index(1)),
            Key.new('b', Key.Index(0), 0, Key.Index(0)),
            Key.new('b', Key.Index(0), 0, Key.Index(1)),
            Key.new('c'),
            Key.new('e'),
        ],
        list(view),
    )

  def test_len(self):
    view = TreeMapView({'a': {'a1': np.array([1, 2])}, 'b': [{0: ['c', 'd']}]})
    self.assertLen(view, 3)

  def test_to_dict(self):
    view = TreeMapView(
        {'a': {'a1': np.array([1, 2])}, 'b': [{0: np.array(['c', 'd'])}]}
    )
    np.testing.assert_equal(
        {
            Key().a.a1: np.array([1, 2]),
            Key.new('b', Key.Index(0), 0): np.array(['c', 'd']),
        },
        view.to_dict(),
    )

  def test_str(self):
    data = {'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}]}
    view = TreeMapView(data)
    self.assertEqual(str(data), str(view))

  def test_copy_and_set_empty(self):
    result = TreeMapView({1: 2}).copy_and_set((), values=())
    self.assertEqual({1: 2}, result.data)

  def test_copy_and_set_single_key(self):
    result = TreeMapView({}).copy_and_set('a', values=1)
    self.assertEqual({'a': 1}, result.data)
    result = TreeMapView([]).copy_and_set(Key.Index(0), values=[1, 2, 3])
    self.assertEqual([[1, 2, 3]], result.data)
    result = TreeMapView({}).copy_and_set(Key.SELF, values=[1, 2, 3])
    self.assertEqual([1, 2, 3], result.data)
    result = TreeMapView({}).copy_and_set(Key(), values=[1, 2, 3])
    self.assertEqual([1, 2, 3], result.data)
    result = TreeMapView({1: 2}).copy_and_set(Key.SKIP, values=[1, 2, 3])
    self.assertEqual({1: 2}, result.data)

  def test_copy_and_set(self):
    data = tree._default_tree(key_path=Key().model1.pred, value=(1, 2, 3))
    result = (
        TreeMapView(data=data)
        .copy_and_set(('a', Key.SKIP), values=(7, 8))
        .copy_and_set(Key(('c', 'b')), values=(7, 8))
        .copy_and_set(Key().model1.pred.at(Key.Index(1)), values=7)
        .copy_and_set(Key().model2.pred3.at(Key.Index(0)), values=[2, 3, 8])
        .copy_and_set('single_key', values=[2, 3, 8])
    ).data
    expected = {
        'model1': {'pred': (1, 7, 3)},
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [[2, 3, 8]]},
        'single_key': [2, 3, 8],
    }
    self.assertEqual(expected, result)

  def test_iterate_with_user_keys(self):
    data = tree._default_tree(key_path=Key().model1.pred, value=(1, 2, 3))
    result = (
        TreeMapView(data=data)
        .copy_and_set(('a', Key.SKIP), values=(7, 8))
        .copy_and_set(Key(('c', 'b')), values=(7, 8))
        .copy_and_set(Key().model1.pred.at(Key.Index(1)), values=7)
        .copy_and_set(Key().model2.pred3.at(Key.Index(0)), values=[2, 3, 8])
        .copy_and_set('single_key', values=[2, 3, 8])
    ).data
    result = TreeMapView.as_view(
        result, key_paths=('a', 'c', 'model1', 'model2', 'single_key')
    )
    expected = (
        7,
        {'b': (7, 8)},
        {'pred': (1, 7, 3)},
        {'pred3': [[2, 3, 8]]},
        [2, 3, 8],
    )
    self.assertEqual(expected, tuple(result.values()))

  def test_copy_and_set_immutable_dict(self):
    # Uses MappingProxyType as an immutable dict for testing.
    data = types.MappingProxyType({'a': 1, 'b': {'c': 2}})
    with self.assertRaisesRegex(TypeError, 'Insert to immutable'):
      TreeMapView(data).copy_and_set(Key.new('b', 'c'), values=0)

  def test_map_fn(self):
    data = dict(a=dict(b=1, c=2), e=3)
    mapped = TreeMapView.as_view(data).copy_and_map(lambda x: x * 2)
    expected = dict(a=dict(b=2, c=4), e=6)
    self.assertEqual(expected, mapped.data)

  def test_view_with_map_fn(self):
    data = dict(a=dict(b=1, c=2), e=3)
    view = TreeMapView.as_view(data, map_fn=lambda x: x * 2)
    mapped = TreeMapView().copy_and_update(view.to_dict())
    expected = dict(a=dict(b=2, c=4), e=6)
    self.assertEqual(expected, mapped.data)

  def test_copy_and_set_with_reserved_keys(self):
    result = (
        TreeMapView({})
        .copy_and_set(Key.SELF, values={'x': {'y': 1}})
        .copy_and_set((Key.SKIP, 'b'), values=(7, 8))
    ).data
    expected = types.MappingProxyType({
        'x': {'y': 1},
        'b': 8,
    })
    self.assertEqual(expected, result)

  def test_copy_and_update_empty(self):
    data = TreeMapView({}).copy_and_set(Key().model1.pred, (1, 2, 3))
    self.assertEqual(data, data.copy_and_update({}))

  def test_copy_and_update_non_empty(self):
    data = TreeMapView({}).copy_and_set(Key().model1.pred, (1, 2, 3))
    new = (
        TreeMapView({})
        .copy_and_set(('a', 'b'), values=(7, 8))
        .copy_and_set(Key.new('c', 'b'), values=(7, 8))
        .copy_and_set(Key().model1.pred.at(Key.Index(0)), values=7)
        .copy_and_set(Key().model2.pred3.at(Key.Index(0)), values=[2, 3, 8])
        .copy_and_set('single_key', values=[2, 3, 8])
    )
    expected = {
        'model1': {'pred': (7, 2, 3)},
        'a': 7,
        'b': 8,
        'c': {'b': [7, 8]},
        'model2': {'pred3': [[2, 3, 8]]},
        'single_key': [2, 3, 8],
    }
    actual = (data | new).data
    self.assertEqual(expected, actual)

  def test_normalize_keys(self):
    self.assertEqual(('a',), tree.normalize_keys('a'))
    self.assertEqual((Key.Index(0),), tree.normalize_keys(Key.Index(0)))
    self.assertEqual((Key(),), tree.normalize_keys(Key()))
    self.assertEqual((Key.SELF,), tree.normalize_keys(Key.SELF))
    self.assertEqual(('a', 'b'), tree.normalize_keys(('a', 'b')))
    self.assertEqual(('a', 'b'), tree.normalize_keys(['a', 'b']))

  def test_tree_shape(self):
    inputs = TreeMapView(
        {
            'a': np.array([1, 2, 3]),
            'b': {'c': np.array([[1, 2], [2, 3]])},
            'c': (np.array([1, 2]), {'e': 6}),
            'd': [1, np.array(2)],
        },
    )
    result = tree.tree_shape(inputs)
    expected = {
        Key().a: (3,),
        Key().b.c: (2, 2),
        Key().c.at(Key.Index(0)): (2,),
        Key().c.at(Key.Index(1)).e: int,
        Key().d.at(Key.Index(0)): int,
        Key().d.at(Key.Index(1)): (),
    }
    self.assertEqual(result, expected)


class ApplyMaskTest(parameterized.TestCase):

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

  @parameterized.named_parameters([
      dict(
          testcase_name='default',
          expected=[1, [5], 5],
      ),
      dict(
          testcase_name='replace_false',
          mask_behavior=tree.MaskBehavior.REPLACE,
          replace_false_with=-1,
          expected=[1, -1, [5, -1], -1, 5],
      ),
  ])
  def test_apply_mask(
      self,
      expected,
      mask_behavior=tree.MaskBehavior.FILTER,
      replace_false_with=None,
  ):
    inputs = [1, 2, np.array([5, 6]), 4, 5]
    mask = [True, False, [True, False], False, True]

    result = tree.apply_mask(
        inputs,
        masks=mask,
        mask_behavior=mask_behavior,
        replace_false_with=replace_false_with,
    )
    self.assert_nested_sequence_equal(result, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='default',
          expected={'a': [1, [5], 5], 'b': [1, 3, 5], 'c': 'irrelevant'},
      ),
      dict(
          testcase_name='replace_false',
          mask_behavior=tree.MaskBehavior.REPLACE,
          expected={
              'a': [1, None, [5, None], None, 5],
              'b': [1, None, 3, None, 5],
              'c': 'irrelevant',
          },
      ),
  ])
  def test_apply_mask_with_dict(
      self, expected, mask_behavior=tree.MaskBehavior.FILTER
  ):
    inputs = {
        'a': [1, 2, np.array([5, 6]), 4, 5],
        'b': [1, 2, 3, 4, 5],
        'c': 'irrelevant',
    }
    mask = {
        'a': [True, False, [True, False], False, True],
        'b': [True, False, True, False, True],
    }
    result = tree.apply_mask(
        inputs,
        masks=mask,
        mask_behavior=mask_behavior,
    )
    self.assert_nested_sequence_equal(result, expected)


if __name__ == '__main__':
  absltest.main()
