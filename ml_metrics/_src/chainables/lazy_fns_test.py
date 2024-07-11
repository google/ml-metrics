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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.chainables import lazy_fns

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
    lazy_fns.picklers.register(custom_pickler)
    self.pickler = lazy_fns.picklers.default

  def test_maybe_make(self):
    self.assertEqual(3, lazy_fns.maybe_make(lazy_fns.trace(len)([1, 2, 3])))
    pickled = self.pickler.dumps(lazy_fns.trace(len)([1, 2, 3]))
    self.assertEqual(3, lazy_fns.maybe_make(pickled))

  @parameterized.named_parameters(
      dict(
          testcase_name='callable',
          lazy_f=lazy_fns.trace_object(Foo(1))(2),
          expected=3,
      ),
      dict(
          testcase_name='member_fn',
          lazy_f=lazy_fns.trace_object(Foo(1)).c(),
          expected=[1, 2, 3],
      ),
      dict(
          testcase_name='member_fn_with_index',
          lazy_f=lazy_fns.trace_object(Foo(1)).c()[1],
          expected=2,
      ),
      dict(
          testcase_name='member_attr',
          lazy_f=lazy_fns.trace_object(Foo(1)).a,
          expected=1,
      ),
      dict(
          testcase_name='pickled',
          lazy_f=pickle.dumps(lazy_fns.trace_object(Foo(1))(2)),
          expected=3,
      ),
  )
  def test_maybe_make_from_traced_object(self, lazy_f, expected):
    self.assertEqual(expected, lazy_fns.maybe_make(lazy_f))

  def test_pickler_register_assertion(self):
    with self.assertRaises(TypeError):
      lazy_fns.picklers.register(len)

  def test_cached_pickle(self):
    pickled_foo = self.pickler.dumps(trace(Foo)(1)(0))
    self.assertEqual(1, maybe_make(pickled_foo))
    with mock.patch.object(self.pickler, 'loads', autospec=True) as mock_loads:
      self.assertEqual(1, maybe_make(pickled_foo))
      mock_loads.assert_not_called()

  def test_maybe_make_cached(self):
    lazy_foo = trace(Foo, use_cache=True)(a=1)
    self.assertEqual(Foo(a=1)(x=1), maybe_make(trace(get)(lazy_foo)(x=1)))
    with mock.patch.object(Foo, '__init__', autospec=True) as mock_cached_make:
      maybe_make(lazy_foo)
      mock_cached_make.assert_not_called()
    self.assertEqual(lazy_fns.cache_info().hits, 1)
    lazy_fns.clear_cache()
    self.assertEqual(lazy_fns.cache_info().hits, 0)

  def test_lazy_fn_not_makeabe_raise_typeerror(self):
    with self.assertRaises(TypeError):
      maybe_make(lazy_fns.LazyFn(fn=3))

  def test_lazy_fn_pickling(self):
    lazy_foo = trace(Foo)(a=trace(get)('a'))
    self.assertEqual(lazy_foo, pickle.loads(pickle.dumps(lazy_foo)))

  def test_lazy_fn_returns_lazy(self):
    lazy_foo = trace(foo)(1, 2, 3, a='a', b='b')
    expected = lazy_fns.LazyFn.new(
        foo,
        args=(1, 2, 3),
        kwargs=dict(a='a', b='b'),
    )
    self.assertEqual(expected, lazy_foo)
    self.assertEqual(foo(1, 2, 3, a='a', b='b'), maybe_make(lazy_foo))

  def test_lazy_fn_with_lazy_args(self):
    lazy_foo = trace(foo)(1, 2, trace(get)(3), a=trace(get)('a'), b='b')
    self.assertEqual(foo(1, 2, 3, a='a', b='b'), maybe_make(lazy_foo))

  def test_lazy_fn_none(self):
    lazy_none = trace(None)()
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
