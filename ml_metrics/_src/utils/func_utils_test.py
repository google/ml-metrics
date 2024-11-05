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
import dataclasses as dc
from absl.testing import absltest
from ml_metrics._src.utils import func_utils


@dc.dataclass()
class Foo:
  a: int
  _b: int = 0
  c: int = 0

  @property
  def b(self):
    return self._b

  @b.setter
  def b(self, value):
    self._b = value

  def __eq__(self, other):
    return self.a == other.a and self.b == other.b and self.c == other.c


class CacheByKwargsTest(absltest.TestCase):

  def test_without_kwargs(self):
    foo_cached = func_utils.lru_cache(settable_kwargs=['b', 'c'])(Foo)
    self.assertEqual(Foo(1, 0, 0), foo_cached(1))
    self.assertEqual(Foo(1, 100, 100), foo_cached(1, b=100, c=100))
    self.assertEqual(foo_cached.cache_info().hits, 1)

  def test_ignore_kwargs(self):
    foo_cached = func_utils.lru_cache(settable_kwargs=['b', '_b', 'c'])(Foo)
    self.assertEqual(Foo(1, 10, 0), foo_cached(1, _b=10))
    self.assertEqual(Foo(1, 100, 100), foo_cached(1, b=100, c=100))
    self.assertEqual(Foo(1, 10, 0), foo_cached(1, b=10))

  def test_cache_partial_kwargs(self):
    foo_cached = func_utils.lru_cache(settable_kwargs=['b'])(Foo)
    self.assertEqual(Foo(1, 0, 10), foo_cached(1, c=10))
    self.assertEqual(Foo(1, 100, 10), foo_cached(1, b=100, c=10))
    self.assertEqual(Foo(1, 10, 10), foo_cached(1, b=10, c=10))

  def test_cache_info(self):
    foo_cached = func_utils.lru_cache(settable_kwargs=['b', 'c'])(Foo)
    foo_cached(1, c=10)
    foo_cached(1, b=100, c=100)
    foo_cached(1, b=10, c=100)
    self.assertEqual(foo_cached.cache_info().hits, 2)
    self.assertEqual(foo_cached.cache_info().misses, 0)
    self.assertEqual(foo_cached.cache_info().currsize, 1)

  def test_cache_insert(self):
    foo_cached = func_utils.lru_cache(settable_kwargs=['b', 'c'])(Foo)
    foo_cached(1)
    self.assertEqual(Foo(1, 0, 1), foo_cached(1, c=1))
    foo_cached(1, c=100, cache_insert_=True)
    self.assertEqual(Foo(1, 0, 100), foo_cached(1))

  def test_attribute_error_raises(self):

    def foo(a, b=1):
      return (a, b)

    foo_cached = func_utils.lru_cache(settable_kwargs=['b'])(foo)
    with self.assertRaises(AttributeError):
      # b is not an attr in the result of foo, thus, cannot uses this kind of
      # caching mechanism by setting the uncached attr afterwards.
      foo_cached(1, b=10)

  def test_cache_clear(self):
    foo_cached = func_utils.lru_cache(settable_kwargs=['b', 'c'])(Foo)
    foo_cached(1)
    foo_cached(1, b=10, c=100)
    foo_cached.cache_clear()
    self.assertEqual(foo_cached.cache_info().hits, 0)
    self.assertEqual(foo_cached.cache_info().misses, 0)
    self.assertEqual(foo_cached.cache_info().currsize, 0)


class LruCacheTest(absltest.TestCase):

  def test_insert_and_get_and_clear(self):
    cache = func_utils.LruCache(maxsize=128)
    cache['a'] = 1
    self.assertEqual(cache['a'], 1)
    self.assertLen(cache, 1)
    self.assertEqual(1, cache.cache_info().currsize)
    self.assertEqual(1, cache.cache_info().hits)
    self.assertEqual(128, cache.cache_info().maxsize)
    cache.cache_clear()
    self.assertEmpty(cache)

  def test_cache_evict(self):
    cache = func_utils.LruCache(maxsize=2)
    cache['a'] = 1
    cache['b'] = 2
    # This pushes 'a' to the latest.
    self.assertEqual(1, cache['a'])
    # This causes 'b' to be evicted.
    cache['c'] = 3
    self.assertEqual(['a', 'c'], list(cache))

  def test_missing_key(self):
    cache = func_utils.LruCache(maxsize=128)
    with self.assertRaises(KeyError):
      _ = cache['a']

  def test_cache_insert(self):
    cache = func_utils.LruCache(maxsize=128)
    cache.cache_insert('a', 1)
    cache.cache_insert('a', 2)
    self.assertEqual(cache['a'], 2)
    self.assertLen(cache, 1)

  def test_cache_iter(self):
    cache = func_utils.LruCache(maxsize=128)
    cache.cache_insert('a', 1)
    cache.cache_insert('b', 2)
    cache.cache_insert('a', 3)
    # The order of the reference here will push 'a' to later than b
    self.assertEqual(cache['b'], 2)
    self.assertEqual(cache['a'], 3)
    self.assertEqual(['b', 'a'], list(cache))


@dc.dataclass(frozen=True)
class SingletonA(metaclass=func_utils.SingletonMeta):
  a: int
  b: str = 'b'


class SingletonMetaTest(absltest.TestCase):

  def test_singleton(self):

    self.assertIs(SingletonA(1, 'b'), SingletonA(1))
    self.assertIs(SingletonA(1), SingletonA(1))
    self.assertIs(SingletonA(1, b='b'), SingletonA(1))
    self.assertIsNot(SingletonA(1, b='b1'), SingletonA(1))
    self.assertIsNot(SingletonA(2), SingletonA(1))


if __name__ == '__main__':
  absltest.main()
