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
from absl.testing import absltest
from ml_metrics._src.utils import func_utils


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


if __name__ == '__main__':
  absltest.main()
