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
import asyncio
import pickle
import queue
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.chainables import lazy_fns
import more_itertools as mit
import numpy as np


trace = lazy_fns.trace
maybe_make = lazy_fns.maybe_make


class CustomPickler:

  def dumps(self, x):
    return pickle.dumps(x)

  def loads(self, x):
    return pickle.loads(x)


class Buildable(str):
  """Test class to simulate a 3rd party lazyfn."""

BuildableAlias = Buildable


def build(x: Buildable):
  return f'build {x}'


def foo(x, *args, a, **kwargs):
  return x, args, a, kwargs


def get(x):
  return x


class Foo:

  def __init__(self, a):
    self.a = a

  def __call__(self, x):
    return self.a + x

  def c(self):
    return [1, 2, 3]


class LazyFnsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    custom_pickler = CustomPickler()
    lazy_fns.pickler.register(custom_pickler)
    self.pickler = lazy_fns.pickler

  def test_cache_lazy_raises(self):
    with self.assertRaises(ValueError):
      lazy_fns.LazyObject.new([1, 2, 3], cache_result=True, lazy_result=True)
    with self.assertRaises(ValueError):
      lazy_fns.trace(len)([1, 2, 3], cache_result_=True, lazy_result_=True)

  @parameterized.named_parameters([
      dict(
          testcase_name='self',
          x=trace(len)([1, 2, 3], lazy_result_=True),
          expected=3,
      ),
      dict(
          testcase_name='member_attr',
          x=trace(Foo)(np.array([1, 2]), lazy_result_=True).a,
          expected=np.array([1, 2]),
      ),
      dict(
          testcase_name='callable',
          x=trace(Foo)(np.array([1, 2]), lazy_result_=True)(2),
          expected=np.array([3, 4]),
      ),
      dict(
          testcase_name='member_attr_after_call',
          x=trace(Foo)(np.array([1, 2]), lazy_result_=True)(2)[0],
          expected=3,
      ),
      dict(
          testcase_name='member_attr_with_index',
          x=trace(Foo)(np.array([1, 2]), lazy_result_=True).a[1],
          expected=2,
      ),
  ])
  def test_lazy_object_lazy_result(self, x, expected):
    lazy_fns.clear_object()
    self.assertIsInstance(x, (lazy_fns.LazyFn, lazy_fns.LazyObject))
    lazy_obj = lazy_fns.maybe_make(x)
    np.testing.assert_array_equal(expected, lazy_fns.maybe_make(lazy_obj))
    self.assertEqual(lazy_fns.object_info().currsize, 1)

  @parameterized.named_parameters([
      dict(testcase_name='int', a=trace(3), b=trace(3)),
      dict(testcase_name='number', a=trace(3.0), b=trace(3)),
      dict(
          testcase_name='int_not_equal_lazy',
          a=lazy_fns.LazyObject.new(10),
          b=trace(10),
          equal=False,
      ),
      dict(
          testcase_name='lazy_not_equal',
          a=lazy_fns.LazyObject.new(10),
          b=lazy_fns.LazyObject.new(10),
          equal=False,
      ),
  ])
  def test_lazy_object_equal(self, a, b, equal=True):
    if equal:
      self.assertEqual(a, b)
    else:
      self.assertNotEqual(a, b)

  def test_lazy_object_iterator(self):

    def foo_genrator(n):
      yield from range(n)

    x = trace(foo_genrator)(3, lazy_result_=True)
    x = lazy_fns.maybe_make(x)
    self.assertIsInstance(x, lazy_fns.LazyObject)
    self.assertIsNone(x.value)
    actual = []
    while (value := lazy_fns.maybe_make(trace(next)(x, None))) is not None:
      actual.append(value)
    self.assertEqual(actual, [0, 1, 2])

  def test_lazy_object_queue(self):

    def foo_q(n):
      q = queue.SimpleQueue()
      for i in range(n):
        q.put(i)
      q.put(None)
      return q

    x = trace(foo_q)(3, lazy_result_=True)
    x = lazy_fns.maybe_make(x)
    self.assertIsInstance(x, lazy_fns.LazyObject)
    actual = []
    while (value := lazy_fns.maybe_make(x.get())) is not None:
      actual.append(value)
    self.assertEqual([0, 1, 2], actual)
    self.assertIsInstance(x.result_(), queue.SimpleQueue)

  @parameterized.named_parameters([
      dict(
          testcase_name='constant',
          x=trace(3),
          expected=3,
      ),
      dict(
          testcase_name='function',
          x=trace(len)([1, 2, 3]),
          expected=3,
      ),
      dict(
          testcase_name='constructor',
          x=lazy_fns.trace(Foo(1))(2),
          expected=3,
      ),
      dict(
          testcase_name='member_fn',
          x=lazy_fns.trace(Foo(1)).c(),
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='member_fn_with_index',
          x=lazy_fns.trace(Foo(1)).c()[1],
          expected=2,
      ),
      dict(
          testcase_name='member_attr',
          x=lazy_fns.trace(Foo(1)).a,
          expected=1,
      ),
      dict(
          testcase_name='member_attr_after_call',
          x=trace(Foo)(np.array([1, 2]))(2)[0],
          expected=3,
      ),
      dict(
          testcase_name='pickled',
          x=pickle.dumps(lazy_fns.trace(Foo(1))(2)),
          expected=3,
      ),
  ])
  def test_maybe_make_from_traced_object(self, x, expected):
    self.assertEqual(expected, lazy_fns.maybe_make(x))

  def test_pickler_register_assertion(self):
    with self.assertRaises(TypeError):
      lazy_fns.pickler.register(len)

  @parameterized.named_parameters([
      dict(testcase_name='len_2', id_len=2),
      dict(testcase_name='len_4', id_len=4),
  ])
  def test_increment_id(self, id_len):
    inc_id = lazy_fns.IncrementId(id_len)
    x = next(inc_id) - inc_id._base
    max_id = 1 << (id_len // 2 * 8)
    # nth starts with 0, max_id - x is off by 1.
    result = mit.nth(inc_id, n=max_id - x - 1)
    self.assertEqual(0, result - inc_id._base)

  def test_maybe_make_cached_normal(self):
    lazy_fns.clear_cache()
    self.assertEqual(lazy_fns.cache_info().hits, 0)
    lazy_foo = trace(Foo)(a=1, cache_result_=True)
    self.assertEqual(Foo(a=1)(x=1), maybe_make(trace(get)(lazy_foo)(x=1)))
    with mock.patch.object(Foo, '__init__', autospec=True) as mock_cached_make:
      maybe_make(lazy_foo)
      mock_cached_make.assert_not_called()
    self.assertEqual(lazy_fns.cache_info().hits, 1)
    lazy_fns.clear_cache()

  def test_maybe_make_cached_only_base_level(self):
    # This is to test that the cache won't propogate beyond the outer layer of
    # the lazy_fn. This is useful to control only part of the function to be
    # cached, e.g., the constuctor of a callable, rather than each individual
    # calls with their args.
    lazy_fns.clear_cache()
    self.assertEqual(lazy_fns.cache_info().hits, 0)
    lazy_foo = trace(Foo)(a=1, cache_result_=True)
    maybe_make(lazy_foo(x=1))
    with mock.patch.object(Foo, '__call__', autospec=True) as mock_cached_make:
      maybe_make(lazy_foo(x=1))
      mock_cached_make.assert_called_once()
    self.assertEqual(lazy_fns.cache_info().hits, 1)

  def test_maybe_make_cached_by_id(self):
    lazy_fns.clear_cache()
    # list is not hashable, hash should fall back by id.
    lazy_foo = trace(Foo)(np.array([1, 2, 3]))(1, cache_result_=True)
    np.testing.assert_array_equal([2, 3, 4], maybe_make(lazy_foo))
    with mock.patch.object(Foo, '__call__', autospec=True) as mock_cached_make:
      maybe_make(lazy_foo)
      mock_cached_make.assert_not_called()
    self.assertEqual(lazy_fns.cache_info().hits, 1)

  def test_lazy_fn_not_makeabe_raise_typeerror(self):
    with self.assertRaises(TypeError):
      maybe_make(lazy_fns.LazyFn(value=3))

  def test_lazy_fn_pickling(self):
    lazy_foo = trace(Foo)(a=trace(get)('a'))
    self.assertEqual(lazy_foo, pickle.loads(pickle.dumps(lazy_foo)))

  def test_lazy_fn_returns_lazy(self):
    lazy_foo = trace(foo)(1, 2, 3, a='a', b='b')
    expected = lazy_fns.LazyFn.new(
        trace(foo),
        args=(1, 2, 3),
        kwargs=dict(a='a', b='b'),
    )
    self.assertEqual(expected, lazy_foo)
    self.assertEqual(foo(1, 2, 3, a='a', b='b'), maybe_make(lazy_foo))

  def test_lazy_fn_with_lazy_args(self):
    lazy_foo = trace(foo)(1, 2, trace(get)(3), a=trace(get)('a'), b='b')
    self.assertEqual(foo(1, 2, 3, a='a', b='b'), maybe_make(lazy_foo))

  def test_lazy_fn_none(self):
    lazy_none = trace(None)
    self.assertIsNone(maybe_make(lazy_none))

  def test_lazy_fn_fn(self):
    lazy_foo = trace(Foo)(a=1)
    self.assertEqual(3, maybe_make(lazy_foo)(2))
    lazy_foo = trace(Foo)(a=1)(2)
    self.assertEqual(3, maybe_make(lazy_foo))
    lazy_iter_fn = trace(lazy_fns.iterate_fn)
    lazy_foo = lazy_iter_fn(trace(Foo)(a=1))
    self.assertEqual([2, 3, 4], maybe_make(lazy_foo)([1, 2, 3]))

  def test_lazy_fn_iter(self):
    lazy_iter_fn = trace(lazy_fns.iterate_fn)
    lazy_foo_class = trace(Foo)
    lazy_foo = lazy_iter_fn(lazy_foo_class(a=1))
    self.assertEqual([2, 3, 4], maybe_make(lazy_foo)([1, 2, 3]))

  def test_async_iter(self):
    lazy_async_iter_fn = trace(lazy_fns.async_iterate_fn)
    lazy_foo_class = trace(Foo)
    lazy_foo = lazy_async_iter_fn(lazy_foo_class(a=1))
    self.assertEqual([2, 3, 4], asyncio.run(maybe_make(lazy_foo)([1, 2, 3])))

  def test_lazy_fn_getattr(self):
    lazy_foo_a = trace(Foo)(a=1).a
    self.assertEqual(1, maybe_make(lazy_foo_a))

  def test_external_makeable(self):
    # Inject external builder.
    lazy_fns.makeables.register(Buildable, build)
    buildable = Buildable('external_buildable')
    self.assertEqual('build external_buildable', maybe_make(buildable))
    lazy_foo = trace(Foo)(a=buildable)
    self.assertEqual(
        'build external_buildable new', maybe_make(lazy_foo)(' new')
    )

    buildable = BuildableAlias('external_buildable')
    self.assertEqual('build external_buildable', maybe_make(buildable))
    lazy_foo = trace(Foo)(a=buildable)
    self.assertEqual(
        'build external_buildable new', maybe_make(lazy_foo)(' new')
    )

  def test_fn_config_to_lazy_fn_direct(self):
    self.assertEqual(
        2,
        lazy_fns.maybe_make(lazy_fns.FnConfig(fn='len', args=[[1, 2]]).make()),
    )

  def test_makeable_lazy_fn(self):
    makeable_foo = lazy_fns.MakeableLazyFn(trace(Foo)(a=1))
    self.assertEqual(3, makeable_foo.make()(x=2))

  def test_lazy_getitem_direct(self):
    self.assertEqual(3, maybe_make(trace(lambda: [1, 2, 3])()[2]))

  def test_lazy_getitem_mixed(self):
    lazy_foo = trace(Foo)(a='a')('b')[1]
    self.assertEqual('b', maybe_make(lazy_foo))


if __name__ == '__main__':
  absltest.main()
