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
"""Utilities for testing, internal use only."""

from collections.abc import Iterable
import time
import unittest

from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.chainables import transform
import numpy as np


class TestSink:

  def __init__(self):
    self.data = []
    self.closed = False

  def write(self, data):
    self.data.append(data)
    return data

  def close(self):
    self.closed = True


class Unpickleable:
  def __call__(self, x):
    return x + 1

  def __getstate__(self):
    raise TypeError('Unpickleable')


def range_with_exc(x, exc_i):
  for i in range(x):
    if i == exc_i:
      raise ValueError(f'range_with_exc at {i}')
    yield i


def range_with_sleep(n, sleep: float = 0):
  for i in range(n):
    if sleep:
      time.sleep(sleep)
    yield i


def range_with_return(n, return_value: int = 0):
  return_value = return_value or n
  for i in range(n):
    yield i
  return return_value


def inf_range(n):
  yield from range(n)
  raise ValueError(
      'The runner does not iterate and copy everything into memory, it should'
      ' never be exhausted.'
  )


class NoLenIter(Iterable):

  def __init__(self, iterable):
    self._iteratable = iterable

  def __len__(self):
    raise ValueError('Cannot call len()')

  def __iter__(self):
    return iter(self._iteratable)


class SequenceWithExc:
  """A range that raises an exception at a specific value."""

  def __init__(self, end: int, exc_i: int, error_type=ValueError):
    self.end = end
    self.exc_i = exc_i
    self.error_type = error_type

  def __len__(self):
    return self.end

  def __getitem__(self, i):
    if i == self.exc_i:
      raise self.error_type(f'SequenceWithExc at {i}')
    return i


def assert_nested_container_equal(
    test: unittest.TestCase, a, b, *, strict=False, places=7
):
  """Asserts that two nested containers are equal."""
  try:
    if strict:
      test.assertEqual(type(a), type(b))
    if isinstance(a, dict) and isinstance(b, dict):
      for (k_a, v_a), (k_b, v_b) in zip(
          sorted(a.items()), sorted(b.items()), strict=True
      ):
        test.assertEqual(k_a, k_b)
        assert_nested_container_equal(
            test, v_a, v_b, strict=strict, places=places
        )
    elif isinstance(a, str) and isinstance(b, str):
      test.assertEqual(a, b)
    elif hasattr(a, '__array__') and hasattr(b, '__array__'):
      np.testing.assert_allclose(a, b)
    elif isinstance(a, Iterable) and isinstance(b, Iterable):
      for a_elem, b_elem in zip(a, b, strict=True):
        assert_nested_container_equal(
            test, a_elem, b_elem, strict=strict, places=places
        )
    else:
      test.assertAlmostEqual(a, b, places=places)
  except Exception as e:  # pylint: disable=broad-except
    test.fail(f'Failed to compare {a} and {b}: {e}')


def sharded_ones(
    total_numbers: int,
    batch_size: int,
    shard_index: int = 0,
    num_shards: int = 1,
):
  num_batches, remainder = divmod(total_numbers, batch_size)
  for i in range(num_batches):
    if i % num_shards == shard_index:
      yield batch_size
  if not shard_index and remainder:
    yield remainder


def sharded_pipeline(
    total_numbers: int,
    batch_size: int,
    shard_index: int = 0,
    num_shards: int = 1,
    fuse_aggregate: bool = True,
    num_threads: int = 0,
):
  """A pipeline to calculate the stats of batches of random integers."""
  data_pipeline = transform.TreeTransform.new(name='datasource').data_source(
      sharded_ones(
          total_numbers,
          batch_size=batch_size,
          shard_index=shard_index,
          num_shards=num_shards,
      )
  )
  apply_pipeline = transform.TreeTransform.new(
      name='apply', num_threads=num_threads
  ).apply(
      fn=lambda batch_size: np.random.randint(1, 100, size=batch_size),
  )

  if fuse_aggregate:
    return data_pipeline.chain(
        apply_pipeline.aggregate(
            output_keys='stats',
            fn=rolling_stats.MeanAndVariance().as_agg_fn(),
        )
    )
  return data_pipeline.interleave(apply_pipeline).interleave(
      transform.TreeTransform.new(name='agg').aggregate(
          output_keys='stats',
          fn=rolling_stats.MeanAndVariance().as_agg_fn(),
      )
  )
