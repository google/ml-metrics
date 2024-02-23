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

from collections.abc import MappingView
import types

from ml_metrics._src.chainables import tree

from absl.testing import absltest

Key = tree.Key
MappingView = tree.MappingView


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


class MappingViewTest(absltest.TestCase):

  def test_as_view(self):
    data = {'a': [1, 2], 'b': [3, 4]}
    view = MappingView.as_view(data)
    self.assertEqual([1, 2], view['a'])
    self.assertEqual(([1, 2], [3, 4]), view['a', 'b'])
    self.assertIsNot(data, view)
    self.assertEqual(view, MappingView.as_view(view))

  def test_get_by_reserved_key(self):
    data = {'a': [1, 2], 'b': [3, 4]}
    view = MappingView.as_view(data)
    self.assertEqual(data, view[Key.SELF])
    self.assertEqual([1, 2], view[Key.new('a', Key.SELF)])

  def test_get_by_skip_raise_error(self):
    with self.assertRaises(KeyError):
      MappingView({'a': 1})[Key.SKIP, 'a']  # pylint: disable=expression-not-assigned

  def test_key_path(self):
    view = MappingView({'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}]})
    self.assertEqual([1, 2], view[Key.new().a.a1])
    self.assertEqual(['c', 'd'], view[Key.new('b', Key.Index(0), 0)])
    self.assertEqual(view.data, view[Key.SELF])
    self.assertEqual(view.data, view[Key()])
    self.assertEqual((), view[()])

  def test_incorrect_keys_raises_error(self):
    # pylint: disable=expression-not-assigned
    with self.assertRaises(ValueError):
      MappingView([1, 2, 3])[0]
    with self.assertRaises(IndexError):
      MappingView([1, 2, 3])[Key.Index(9)]
    with self.assertRaises(KeyError):
      MappingView({'a': 1})['b']
    with self.assertRaises(TypeError):
      MappingView({'a': 1})[set('a')]
    # pylint: enable=expression-not-assigned

  def test_iter(self):
    data = {'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}]}
    view = MappingView(data)
    self.assertEqual(
        [Key.new('a', 'a1'), Key.new('b', Key.Index(0), 0)], list(view)
    )
    view = MappingView(data, ignore_leaf_sequence=False)
    self.assertEqual(
        [
            Key.new('a', 'a1', Key.Index(0)),
            Key.new('a', 'a1', Key.Index(1)),
            Key.new('b', Key.Index(0), 0, Key.Index(0)),
            Key.new('b', Key.Index(0), 0, Key.Index(1)),
        ],
        list(view),
    )

  def test_len(self):
    view = MappingView({'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}]})
    self.assertLen(view, 2)

  def test_to_dict(self):
    view = MappingView({'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}]})
    self.assertEqual(
        {Key().a.a1: [1, 2], Key.new('b', Key.Index(0), 0): ['c', 'd']},
        view.to_dict(),
    )

  def test_str(self):
    data = {'a': {'a1': [1, 2]}, 'b': [{0: ['c', 'd']}]}
    view = MappingView(data)
    self.assertEqual(str(data), str(view))

  def test_copy_and_set_empty(self):
    result = MappingView({1: 2}).copy_and_set((), values=())
    self.assertEqual({1: 2}, result.data)

  def test_copy_and_set_single_key(self):
    result = MappingView({}).copy_and_set('a', values=1)
    self.assertEqual({'a': 1}, result.data)
    result = MappingView([]).copy_and_set(Key.Index(0), values=[1, 2, 3])
    self.assertEqual([[1, 2, 3]], result.data)
    result = MappingView({}).copy_and_set(Key.SELF, values=[1, 2, 3])
    self.assertEqual([1, 2, 3], result.data)
    result = MappingView({}).copy_and_set(Key(), values=[1, 2, 3])
    self.assertEqual([1, 2, 3], result.data)
    result = MappingView({1: 2}).copy_and_set(Key.SKIP, values=[1, 2, 3])
    self.assertEqual({1: 2}, result.data)

  def test_copy_and_set(self):
    data = tree._default_tree(key_path=Key().model1.pred, value=(1, 2, 3))
    # Uses MappingProxyType as an immutable dict for testing.
    data = types.MappingProxyType(data)
    result = (
        tree.MappingView(data=data, consistent_type=True)
        .copy_and_set(('a', Key.SKIP), values=(7, 8))
        .copy_and_set(Key(('c', 'b')), values=(7, 8))
        .copy_and_set(Key().model1.pred.at(Key.Index(1)), values=7)
        .copy_and_set(Key().model2.pred3.at(Key.Index(0)), values=[2, 3, 8])
        .copy_and_set('single_key', values=[2, 3, 8])
    ).data
    expected = types.MappingProxyType({
        'model1': {'pred': (1, 7, 3)},
        'a': 7,
        'c': {'b': (7, 8)},
        'model2': {'pred3': [[2, 3, 8]]},
        'single_key': [2, 3, 8],
    })
    self.assertEqual(expected, result)
    self.assertIsInstance(result, types.MappingProxyType)

  def test_copy_and_set_with_reserved_keys(self):
    result = (
        tree.MappingView({})
        .copy_and_set(Key.SELF, values={'x': {'y': 1}})
        .copy_and_set((Key.SKIP, 'b'), values=(7, 8))
    ).data
    expected = types.MappingProxyType({
        'x': {'y': 1},
        'b': 8,
    })
    self.assertEqual(expected, result)

  def test_copy_and_update_empty(self):
    data = tree.MappingView({}).copy_and_set(Key().model1.pred, (1, 2, 3))
    self.assertEqual(data, data.copy_and_update({}))

  def test_copy_and_update_non_empty(self):
    data = tree.MappingView({}).copy_and_set(Key().model1.pred, (1, 2, 3))
    new = (
        tree.MappingView({})
        .copy_and_set(('a', 'b'), values=(7, 8))
        .copy_and_set(Key(('c', 'b')), values=(7, 8))
        .copy_and_set(Key().model1.pred.at(Key.Index(0)), values=7)
        .copy_and_set(Key().model2.pred3.at(Key.Index(0)), values=[2, 3, 8])
        .copy_and_set('single_key', values=[2, 3, 8])
    )
    expected = {
        'model1': {'pred': [7]},
        'a': 7,
        'b': 8,
        'c': {'b': (7, 8)},
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


if __name__ == '__main__':
  absltest.main()
