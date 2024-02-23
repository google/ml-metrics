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

from ml_metrics._src.chainables import lazy_fns

from absl.testing import absltest

trace = lazy_fns.trace
maybe_make = lazy_fns.maybe_make


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


class LazyFnsTest(absltest.TestCase):

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
if __name__ == '__main__':
  absltest.main()
