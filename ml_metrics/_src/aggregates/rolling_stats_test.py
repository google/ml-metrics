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
import dataclasses
import math

from absl.testing import parameterized
from ml_metrics._src.aggregates import rolling_stats
from ml_metrics._src.chainables import transform
from ml_metrics._src.utils import test_utils
import more_itertools as mit
import numpy as np

from absl.testing import absltest


class HistogramTest(parameterized.TestCase):

  def test_histogram_merge(self):
    bin_range = (0, 1)
    bins = 5

    input_1 = (0, 1, 0, 1, 1, 1, 0, 1)
    input_2 = (0.2, 0.8, 0.5, -0.1, 0.5, 0.8, 0.2, 1.1)

    histogram_1 = rolling_stats.Histogram(range=bin_range, bins=bins).add(
        input_1
    )
    histogram_2 = rolling_stats.Histogram(range=bin_range, bins=bins).add(
        input_2
    )

    actual_histogram = histogram_1.merge(histogram_2)
    result = actual_histogram.result()

    expected_histogram = (
        # Input values in each bin:
        3,  # (0, 0, 0)
        2,  # (0.2, 0.2)
        2,  # (0.5, 0.5)
        0,  # (),
        7,  # (0.8, 0.8, 1, 1, 1, 1, 1)
    )
    expected_bin_edges = (0, 0.2, 0.4, 0.6, 0.8, 1)

    self.assertEqual(actual_histogram.bins, 5)
    self.assertSequenceEqual(actual_histogram.range, bin_range)

    np.testing.assert_equal(
        actual_histogram.hist, expected_histogram
    )
    np.testing.assert_equal(result.hist, expected_histogram)

    np.testing.assert_allclose(
        actual_histogram.bin_edges, expected_bin_edges
    )
    np.testing.assert_allclose(result.bin_edges, expected_bin_edges)

  def test_histogram_simple(self):
    bin_range = (0, 1)
    bins = 5

    input_1 = (0, 1, 0, 1, 1, 1, 0, 1)
    input_2 = (0.2, 0.8, 0.5, -0.1, 0.5, 0.8, 0.2, 1.1)

    histogram = rolling_stats.Histogram(range=bin_range, bins=bins)
    histogram.add(input_1)
    histogram.add(input_2)
    actual = histogram.result()

    expected_histogram = (
        # Input values in each bin:
        3,  # (0, 0, 0)
        2,  # (0.2, 0.2)
        2,  # (0.5, 0.5)
        0,  # (),
        7,  # (0.8, 0.8, 1, 1, 1, 1, 1)
    )
    expected_bin_edges = (0, 0.2, 0.4, 0.6, 0.8, 1)

    np.testing.assert_equal(actual.hist, expected_histogram)
    np.testing.assert_allclose(actual.bin_edges, expected_bin_edges)

  def test_histogram_simple_with_bin_edges(self):
    bins = (0, 0.2, 0.4, 0.6, 0.8, 1)
    input_1 = (0, 1, 0, 1, 1, 1, 0, 1)
    input_2 = (0.2, 0.8, 0.5, -0.1, 0.5, 0.8, 0.2, 1.1)
    histogram = rolling_stats.Histogram(bins=bins)
    histogram.add(input_1)
    histogram.add(input_2)
    actual = histogram.result()
    expected_histogram = (
        # Input values in each bin:
        3,  # (0, 0, 0)
        2,  # (0.2, 0.2)
        2,  # (0.5, 0.5)
        0,  # (),
        7,  # (0.8, 0.8, 1, 1, 1, 1, 1)
    )
    np.testing.assert_equal(actual.hist, expected_histogram)

  def test_histogram_one_large_batch(self):
    np.random.seed(seed=0)

    num_values = 1000000
    bins = 10
    left_boundary = -1e6
    right_boundary = 1e6

    inputs = np.random.uniform(
        low=left_boundary, high=right_boundary, size=num_values
    )

    hist_fn = rolling_stats.Histogram(
        range=(left_boundary, right_boundary), bins=bins
    )
    actual_result = hist_fn.as_agg_fn()(inputs)
    expected_histogram, expected_bin_edges = np.histogram(
        inputs, bins=bins, range=(left_boundary, right_boundary)
    )

    np.testing.assert_equal(actual_result.hist, expected_histogram)
    np.testing.assert_equal(actual_result.bin_edges, expected_bin_edges)

  def test_histogram_many_large_batches(self):
    np.random.seed(seed=0)

    batches = 1000
    batch_size = 1000
    bins = 10
    left_boundary = -1e6
    right_boundary = 1e6

    inputs = np.random.uniform(
        low=left_boundary, high=right_boundary, size=(batches, batch_size)
    )

    state = rolling_stats.Histogram(
        range=(left_boundary, right_boundary),
        bins=bins,
    )
    for test_values in inputs:
      state.add(test_values)
    actual_result = state.result()

    expected_histogram, expected_bin_edges = np.histogram(
        inputs, bins=bins, range=(left_boundary, right_boundary)
    )

    np.testing.assert_equal(actual_result.hist, expected_histogram)
    np.testing.assert_equal(actual_result.bin_edges, expected_bin_edges)

  @parameterized.named_parameters(
      [
          dict(
              testcase_name='same_bin_edges_shape',
              bin_edges_1=(0, 0.25, 0.5, 0.75, 1.0),
              bin_edges_2=(0, 0.2, 0.5, 0.8, 1.0),
          ),
          dict(
              testcase_name='different_bin_edges_shapes',
              bin_edges_1=(0, 0.25, 0.5, 0.75, 1.0),
              bin_edges_2=(0, 0.5, 1.0),
          ),
      ],
  )
  def test_histogram_invalid_bin_edges(self, bin_edges_1, bin_edges_2):
    place_holder_range = (0, 1)
    place_holder_hist = np.array((1, 1, 1, 1))

    hist_1 = rolling_stats.Histogram(
        range=place_holder_range,
        bins=4,
        _hist=place_holder_hist,
        _bin_edges=np.array(bin_edges_1),
    )

    hist_2 = dataclasses.replace(hist_1, _bin_edges=np.array(bin_edges_2))
    with self.assertRaisesRegex(
        ValueError,
        'The bin edges of the two Histograms must be equal, but recieved'
    ):
      hist_1.merge(hist_2)


class CounterTest(absltest.TestCase):

  def test_counter_call(self):
    counter = rolling_stats.Counter()
    result = counter(['a', 'b'])
    self.assertEqual({'a': 1, 'b': 1}, result)

  def test_counter_agg_fn_call(self):
    counter = rolling_stats.Counter().as_agg_fn()
    result = counter(['a', 'b'])
    self.assertEqual({'a': 1, 'b': 1}, result)

  def test_counter_update(self):
    counter = rolling_stats.Counter()
    for x in ['a b'.split(' '), 'c a'.split(' ')]:
      counter.add(x)
    self.assertEqual({'a': 2, 'b': 1, 'c': 1}, counter.result())

  def test_counter_merge(self):
    counter_1 = rolling_stats.Counter()
    counter_2 = rolling_stats.Counter()
    counter_2.add(['a', 'b'])
    counter_1.add(['a', 'b'])
    counter_1.merge(counter_2)
    self.assertEqual({'a': 2, 'b': 2}, counter_1.result())


def get_expected_mean_and_variance(batches, batch_score_fn=None):
  if batch_score_fn is not None:
    batches = [batch_score_fn(batch) for batch in batches]
  batches = np.asarray(batches)
  batch = batches.reshape(-1, *batches.shape[2:])
  return rolling_stats.MeanAndVariance(
      batch_score_fn=batch_score_fn,
      _mean=np.nanmean(batch, axis=0),
      _var=np.nanvar(batch, axis=0),
      _count=np.nansum(~np.isnan(batch), axis=0),
      _input_shape=batches.shape[1:] if batch.size else (),
  )


class UnboundesSamplerTest(absltest.TestCase):

  def test_default_call(self):
    sampler = rolling_stats.UnboundedSampler()
    result = sampler(['a', 'b'])
    self.assertEqual(['a', 'b'], result)

  def test_as_agg_fn_call(self):
    sampler = rolling_stats.UnboundedSampler().as_agg_fn()
    result = sampler(['a', 'b'])
    self.assertEqual(['a', 'b'], result)

  def test_batch_update(self):
    sampler = rolling_stats.UnboundedSampler()
    sampler.add(['a', 'b'])
    sampler.add(['c', 'd'])
    self.assertEqual(['a', 'b', 'c', 'd'], sampler.result())

  def test_merge(self):
    sampler_1 = rolling_stats.UnboundedSampler()
    sampler_2 = rolling_stats.UnboundedSampler()
    sampler_1.add(['a', 'b'])
    sampler_2.add(['c', 'b'])
    sampler_1.merge(sampler_2)
    self.assertEqual(['a', 'b', 'c', 'b'], sampler_1.result())

  def test_with_transform_without_input_keys(self):
    t = transform.TreeTransform().agg(
        fn=rolling_stats.UnboundedSampler().as_agg_fn(),
        output_keys='samples',
    )
    it = t.make().iterate([['a', 'b'], ['c']])
    _ = mit.last(it)
    metric_key = transform.MetricKey(metrics=('samples',))
    expected_state = rolling_stats.UnboundedSampler(
        _samples=(['a', 'b', 'c'],), _multi_input=False
    )
    self.assertEqual(expected_state, it.agg_state[metric_key])
    self.assertEqual(['a', 'b', 'c'], it.agg_result['samples'])

  def test_with_transform_with_input_keys(self):
    t = transform.TreeTransform().agg(
        fn=rolling_stats.UnboundedSampler().as_agg_fn(),
        output_keys=('a', 'b'),
        input_keys=('a', 'b'),
    )
    inputs = [{'a': [0, 1], 'b': [10]}, {'a': [2, 3], 'b': [20]}]
    it = t.make().iterate(inputs)
    _ = mit.last(it)
    self.assertEqual([0, 1, 2, 3], it.agg_result['a'])
    self.assertEqual([10, 20], it.agg_result['b'])


class FixedSizeSampleTest(parameterized.TestCase):
  def _assert_fixed_size_samples_equal(self, actual, expected):
    self.assertEqual(actual.max_size, expected.max_size)
    self.assertEqual(actual.seed, expected.seed)
    self.assertSameElements(actual._reservoir, expected._reservoir)
    self.assertEqual(
        actual._num_samples_reviewed, expected._num_samples_reviewed
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='both_reservoirs_full',
          reservoir_original=[1, 2, 3, 4, 5],
          reservoir_other=[6, 7, 8, 9, 10],
          num_samples_original=5,
          num_samples_other=5,
          expected_reservoir=(2, 1, 9, 10, 7),
          expected_num_samples_reviewed=10,
      ),
      dict(
          testcase_name='only_original_reservoir_full',
          reservoir_original=[1, 2, 3, 4, 5],
          reservoir_other=[6, 7],
          num_samples_original=5,
          num_samples_other=2,
          expected_reservoir=(2, 1, 7, 5, 6),
          expected_num_samples_reviewed=7,
      ),
      dict(
          testcase_name='only_other_reservoir_full',
          reservoir_original=[1, 2, 3],
          reservoir_other=[6, 7, 8, 9, 10],
          num_samples_original=3,
          num_samples_other=5,
          expected_reservoir=(1, 2, 9, 10, 7),
          expected_num_samples_reviewed=8,
      ),
      dict(
          testcase_name='neither_reservoir_full',
          reservoir_original=[1, 2],
          reservoir_other=[6, 7, 8],
          num_samples_original=2,
          num_samples_other=3,
          expected_reservoir=(1, 2, 6, 7, 8),
          expected_num_samples_reviewed=5,
      ),
      dict(
          testcase_name='merged_reservoir_not_full',
          reservoir_original=[1, 2],
          reservoir_other=[6, 7],
          num_samples_original=2,
          num_samples_other=2,
          expected_reservoir=(1, 2, 6, 7),
          expected_num_samples_reviewed=4,
      ),
      dict(
          testcase_name='different_num_samples',
          reservoir_original=[1, 2, 3, 4, 5],
          reservoir_other=[6, 7, 8, 9, 10],
          num_samples_original=11,
          num_samples_other=7,
          expected_reservoir=(2, 1, 9, 10, 7),
          expected_num_samples_reviewed=18,
      ),
  )
  def test_fixed_size_sample_merge(
      self,
      reservoir_original,
      reservoir_other,
      num_samples_original,
      num_samples_other,
      expected_reservoir,
      expected_num_samples_reviewed,
  ):
    original = rolling_stats.FixedSizeSample(
        max_size=5,
        seed=0,
        _reservoir=reservoir_original,
        _num_samples_reviewed=num_samples_original,
    )
    other = rolling_stats.FixedSizeSample(
        max_size=5,
        seed=0,
        _reservoir=reservoir_other,
        _num_samples_reviewed=num_samples_other,
    )
    original.merge(other)
    expected = rolling_stats.FixedSizeSample(
        max_size=5,
        seed=0,
        _reservoir=expected_reservoir,
        _num_samples_reviewed=expected_num_samples_reviewed,
    )
    self._assert_fixed_size_samples_equal(expected, original)

  def test_fixed_size_sample_merge_different_max_sizes(self):
    res_1 = [1, 2, 3, 4, 5]
    res_2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    sampler = rolling_stats.FixedSizeSample(
        max_size=5,
        seed=0,
        _reservoir=res_1,
        _num_samples_reviewed=10,
    )
    other = rolling_stats.FixedSizeSample(
        max_size=10, seed=0, _reservoir=res_2, _num_samples_reviewed=10
    )
    sampler.merge(other)
    expected_result = (2, 1, 70, 100, 60)
    np.testing.assert_array_equal(sampler.result(), expected_result)

  def test_fixed_size_sample_merge_smaller_samples_than_max_size(self):
    sampler = rolling_stats.FixedSizeSample(
        max_size=5,
        seed=0,
        _reservoir=[1, 2],
        _num_samples_reviewed=2,
    )
    other = rolling_stats.FixedSizeSample(
        max_size=10, seed=0, _reservoir=[10], _num_samples_reviewed=1
    )
    sampler.merge(other)
    expected_result = (1, 2, 10)
    np.testing.assert_array_equal(sampler.result(), expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='random_seed_0',
          random_seed=0,
          expected_reservoir=(10, 2, 9, 4, 5, 6, 7, 8),
      ),
      dict(
          testcase_name='random_seed_1',
          random_seed=1,
          expected_reservoir=(9, 10, 3, 4, 5, 6, 7, 8),
      ),
      dict(
          testcase_name='random_seed_2',
          random_seed=2,
          expected_reservoir=(1, 2, 3, 9, 5, 6, 10, 8),
      ),
      dict(
          testcase_name='random_seed_3',
          random_seed=3,
          expected_reservoir=(1, 10, 3, 4, 5, 6, 7, 8),
      ),
      dict(
          testcase_name='random_seed_4',
          random_seed=4,
          expected_reservoir=(1, 2, 3, 4, 5, 6, 7, 10),
      ),
      dict(
          testcase_name='random_seed_5',
          random_seed=5,
          expected_reservoir=(1, 2, 3, 9, 5, 6, 7, 8),
      ),
      dict(
          testcase_name='random_seed_6',
          random_seed=6,
          expected_reservoir=(1, 2, 10, 4, 5, 6, 7, 9),
      ),
      dict(
          testcase_name='random_seed_7',
          random_seed=7,
          expected_reservoir=(1, 2, 3, 4, 9, 6, 10, 8),
      ),
      dict(
          testcase_name='random_seed_8',
          random_seed=8,
          expected_reservoir=(1, 9, 10, 4, 5, 6, 7, 8),
      ),
      dict(
          testcase_name='random_seed_9',
          random_seed=9,
          expected_reservoir=(9, 2, 3, 4, 10, 6, 7, 8),
      ),
      dict(
          testcase_name='random_seed_10',
          random_seed=10,
          expected_reservoir=(1, 2, 3, 4, 5, 6, 10, 8),
      ),
  )
  def test_fixed_size_sample_random_seed(self, random_seed, expected_reservoir):
    stream = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    sampler = rolling_stats.FixedSizeSample(max_size=8, seed=random_seed)
    sampler.add(stream)
    np.testing.assert_array_equal(expected_reservoir, sampler.result())

  def test_fixed_size_sample_str(self):
    stream = (
        'Apple',
        'Banana',
        'Cherry',
        'Durian',
        'Elderberry',
        'Fig',
        'Grape',
        'Honeydew',
        'Jackfruit',
        'Kiwi',
    )
    sampler = rolling_stats.FixedSizeSample(max_size=4, seed=0)
    sampler.add(stream)
    expected_reservoir = ('Fig', 'Elderberry', 'Jackfruit', 'Honeydew')
    np.testing.assert_array_equal(expected_reservoir, sampler.result())

  def test_fixed_size_sample_one_large_batch(self):
    batch_size = 1000000
    datastream = np.random.default_rng(0).uniform(
        low=-1e6, high=1e6, size=batch_size
    )
    sampler = rolling_stats.FixedSizeSample(max_size=10, seed=0)
    sampler.add(datastream)
    expected = (
        780105.0592725971,
        278814.6430954598,
        -630990.0367278787,
        246931.43730201898,
        641081.8690512367,
        -440344.3624372452,
        306002.2913993879,
        70767.14614483598,
        974427.8183713104,
        -999283.5926202195,
    )
    self.assertSequenceAlmostEqual(expected, sampler.result(), places=9)

  def test_fixed_size_sample_many_batches(self):
    size = (1000, 1000)
    datastreams = np.random.default_rng(0).uniform(
        low=-1e6, high=1e6, size=size
    )

    metric = rolling_stats.FixedSizeSample(max_size=10, seed=0)
    for datastream in datastreams:
      metric.add(datastream)

    expected_reservoir = (
        -782795.1778994582,
        -807734.1179696717,
        -87551.47009099205,
        -400659.55050634814,
        192581.4211200222,
        -228024.2926574516,
        -640421.47726782,
        638141.0141386185,
        612068.7958296072,
        958226.8426046742,
    )
    self.assertSequenceAlmostEqual(
        metric.result(), expected_reservoir, places=9
    )

  def test_sampling_add_uniformness(self):
    # There is a 1/10 chance that each value is sampled.
    max_range, max_size = 100, 10
    actual_counter = np.zeros(max_range)
    num_runs = 1000
    for i in range(num_runs):
      sampler = rolling_stats.FixedSizeSample(max_size=max_size, seed=i)
      for batch in mit.batched(np.arange(max_range), 9):
        sampler.add(batch)
      for v in sampler.result():
        actual_counter[v] += 1
    actual_counter /= num_runs
    np.testing.assert_array_less(actual_counter - max_size / max_range, 0.03)

  def test_sampling_merge_uniformness(self):
    # There is a 1/10 chance that each value is sampled.
    max_range, max_size = 100, 10
    actual_counter = np.zeros(max_range)
    num_runs = 1000
    for i in range(num_runs):
      sampler = rolling_stats.FixedSizeSample(max_size=max_size, seed=i)
      for batch in mit.batched(np.arange(max_range), 9):
        other = rolling_stats.FixedSizeSample(max_size=max_size, seed=i)
        other.add(batch)
        sampler.merge(other)
      for v in sampler.result():
        actual_counter[v] += 1
    actual_counter /= num_runs
    np.testing.assert_array_less(actual_counter - max_size / max_range, 0.03)

  def test_as_agg_fn(self):
    sampler = rolling_stats.FixedSizeSample(max_size=5, seed=0)
    sampler_agg_fn = sampler.as_agg_fn()
    actual = sampler_agg_fn(np.arange(10))
    sampler.add(np.arange(10))
    self.assertEqual(sampler.result(), actual)


class MeanAndVarianceTest(parameterized.TestCase):

  def assertDataclassAlmostEqual(
      self,
      expected: 'dataclasses.DataclassInstance',
      got: 'dataclasses.DataclassInstance',
  ):
    """Helper function for using assertAlmostEquals in dictionaries."""
    expected = dataclasses.asdict(expected)
    got = dataclasses.asdict(got)
    self.assertEqual(expected.keys(), got.keys())
    for key, value in expected.items():
      try:
        if key == 'batch_score_fn':
          self.assertAlmostEqual(value, got[key])
        elif key == '_input_shape':
          np.testing.assert_array_equal(value[1:], got[key][1:])
        else:
          np.testing.assert_allclose(value, got[key])
      except AssertionError:
        self.fail(f'Failed to assert {key}: {value} == {got[key]}')

  def test_count_normal(self):
    self.assertEqual(3, rolling_stats.Count()(['a', 'b', 'c']))

  def test_count_merge(self):
    count1 = rolling_stats.Count()
    count2 = rolling_stats.Count()
    count1.add(['a', 'b', 'c'])
    count2.add(['a', 'b', 'c'])
    count1.merge(count2)
    self.assertEqual(6, count1.result())

  def test_count_multi_column_inputs(self):
    # Simulate two column of inputs that requires a batch_score_fn to process.
    batches = [(1, 1), (2, 2)]
    count = rolling_stats.Count(batch_score_fn=lambda *batch: list(batch))
    for batch in batches:
      count.add(*batch)
    self.assertEqual(4, count.result())

  def test_count_as_agg_fn(self):
    p = (
        transform.TreeTransform()
        .batch(2)
        .agg(rolling_stats.Count().as_agg_fn())
    )
    # auto-batch for single element
    self.assertEqual(3, p.make()(input_iterator=range(3)))

  def test_count_str(self):
    count = rolling_stats.Count()
    count.add(['a', 'b', 'c'])
    self.assertEqual('count: 3', str(count))

  def test_count_multi_column_no_batch_score_fn_raise(self):
    count = rolling_stats.Count()
    with self.assertRaisesRegex(ValueError, 'inputs requires a batch_score_fn'):
      count.add(1, 2)

  def test_mean_normal(self):
    # This only tests that Mean().get_result() returns the mean value directly.
    state = rolling_stats.Mean().add([1, 2, 3])
    self.assertIsInstance(state.result(), float)
    self.assertEqual(2.0, state.result())
    state.add([4, 5, 6])
    result = state.result()
    self.assertIsInstance(result, float)
    self.assertIsInstance(state.count, np.int64)
    self.assertIsInstance(state.total, float)
    self.assertEqual(3.5, result)
    self.assertEqual('mean: 3.5', str(state))

  def test_var_normal(self):
    # This only tests that Var().get_result() returns the variance directly.
    state = rolling_stats.Var().add([1, 1, 1])
    self.assertIsInstance(state.result(), float)
    self.assertEqual(0, state.result())
    state.add([1, 1, 1])
    result = state.result()
    self.assertIsInstance(result, float)
    self.assertEqual(0, result)
    self.assertEqual('var: 0.0', str(state))

  @parameterized.named_parameters([
      dict(
          testcase_name='1_batch_1_element_1_dim',
          num_batch=1,
          num_elements_per_batch=1,
          num_dimension=1,
      ),
      dict(
          testcase_name='1_batch_1_element_10_dim',
          num_batch=1,
          num_elements_per_batch=1,
          num_dimension=10,
      ),
      dict(
          testcase_name='1_batch_10_element_1_dim',
          num_batch=1,
          num_elements_per_batch=10,
          num_dimension=1,
      ),
      dict(
          testcase_name='1_batch_10_element_10_dim',
          num_batch=1,
          num_elements_per_batch=10,
          num_dimension=10,
      ),
      dict(
          testcase_name='10_batch_1_element_1_dim',
          num_batch=10,
          num_elements_per_batch=1,
          num_dimension=1,
      ),
      dict(
          testcase_name='10_batch_1_element_10_dim',
          num_batch=10,
          num_elements_per_batch=1,
          num_dimension=10,
      ),
      dict(
          testcase_name='10_batch_10_element_1_dim',
          num_batch=10,
          num_elements_per_batch=10,
          num_dimension=1,
      ),
      dict(
          testcase_name='10_batch_10_element_10_dim',
          num_batch=10,
          num_elements_per_batch=10,
          num_dimension=10,
      ),
      dict(
          testcase_name='1000_batch_1_element_1_dim',
          num_batch=1000,
          num_elements_per_batch=1,
          num_dimension=1,
      ),
      dict(
          testcase_name='1000_batch_1000_element_1_dim',
          num_batch=1000,
          num_elements_per_batch=1000,
          num_dimension=1,
      ),
  ])
  def test_mean_and_variance(
      self, num_batch, num_elements_per_batch, num_dimension
  ):
    batches = np.random.randn(num_batch, num_elements_per_batch, num_dimension)
    state = rolling_stats.MeanAndVariance()

    last_batch_result = None
    for batch in batches:
      last_batch_result = state.add(batch)
    self.assertIsInstance(last_batch_result, rolling_stats.MeanAndVariance)

    expected_last_batch_result = get_expected_mean_and_variance(batches[-1:])
    self.assertDataclassAlmostEqual(
        expected_last_batch_result, last_batch_result
    )

    expected_state = get_expected_mean_and_variance(batches)
    self.assertDataclassAlmostEqual(expected_state, state)

  def test_mean_and_variance_with_batch_score_fn(self):
    batch = [
        dict(score1=1, score2=2, score3=3),
        dict(score1=4, score2=5, score3=6),
        dict(score1=7, score2=8, score3=9),
    ]
    batch_score_fn = lambda batch: [
        [x['score1'], x['score2'], x['score3']] for x in batch
    ]
    state = rolling_stats.MeanAndVariance(batch_score_fn=batch_score_fn)
    state.add(batch)
    expected_state = get_expected_mean_and_variance([batch], batch_score_fn)
    self.assertDataclassAlmostEqual(expected_state, state)

  def test_mean_and_variance_multi_dimension_with_nan(self):
    # Create batches with shape (2, 3, 4)
    batches = np.asarray([
        [
            [1, 2, 3, np.nan],
            [np.nan, 5, 6, np.nan],
            [7, 8, 9, np.nan],
        ],
        [
            [np.nan, 11, 12, np.nan],
            [13, np.nan, 15, np.nan],
            [16, 17, 18, np.nan],
        ],
    ])

    state = rolling_stats.MeanAndVariance()

    last_batch_result = None
    for batch in batches:
      last_batch_result = state.add(batch)
    self.assertIsInstance(last_batch_result, rolling_stats.MeanAndVariance)

    expected_last_batch_result = get_expected_mean_and_variance(batches[-1:])
    self.assertDataclassAlmostEqual(
        expected_last_batch_result, last_batch_result
    )

    expected_state = get_expected_mean_and_variance(batches)
    self.assertDataclassAlmostEqual(expected_state, state)

  def test_mean_and_variance_add_with_incompatible_shape(self):
    batch = np.array([[1, 2], [2, 3]])
    state = rolling_stats.MeanAndVariance()
    state.add(batch)
    with self.assertRaisesRegex(ValueError, 'Incompatible shape'):
      state.add(np.array([1, 2]))

  def test_mean_and_variance_merge_with_incompatible_shape(self):
    batch1 = np.array([[1, 2], [2, 3]])
    batch2 = np.array([1, 2])
    state1 = rolling_stats.MeanAndVariance().add(batch1)
    state2 = rolling_stats.MeanAndVariance().add(batch2)
    with self.assertRaisesRegex(ValueError, 'Incompatible shape'):
      state1.merge(state2)

  @parameterized.named_parameters([
      dict(
          testcase_name='all_nan',
          partial_nan=False,
      ),
      dict(
          testcase_name='partial_nan',
          partial_nan=True,
      ),
  ])
  def test_mean_and_variance_batch_with_nan(self, partial_nan):
    batch = [np.nan] * 3 + [1, 2, 3] * int(partial_nan)
    state = rolling_stats.MeanAndVariance()
    state.add(batch)

    expected_state = get_expected_mean_and_variance(batch)
    self.assertDataclassAlmostEqual(expected_state, state)

  @parameterized.named_parameters([
      dict(
          testcase_name='nonempty_states',
          is_self_empty=False,
          is_other_empty=False,
      ),
      dict(
          testcase_name='nonempty_with_empty_state',
          is_self_empty=False,
          is_other_empty=True,
      ),
      dict(
          testcase_name='empty_with_nonempty_state',
          is_self_empty=True,
          is_other_empty=False,
      ),
      dict(
          testcase_name='empty_states',
          is_self_empty=True,
          is_other_empty=True,
      ),
      dict(
          testcase_name='nonempty_multi_dim_states',
          is_self_empty=True,
          is_other_empty=True,
          num_dimension=10,
      ),
  ])
  def test_mean_and_variance_merge(
      self, is_self_empty, is_other_empty, num_dimension=1
  ):
    self_batch = np.random.randn(int(not is_self_empty), 30, num_dimension)
    other_batch = np.random.randn(int(not is_other_empty), 30, num_dimension)
    self_state = rolling_stats.MeanAndVariance()
    other_state = rolling_stats.MeanAndVariance()

    for batch in self_batch:
      self_state.add(batch)

    for batch in other_batch:
      other_state.add(batch)

    self_state.merge(other_state)

    expected_state = get_expected_mean_and_variance(
        np.append(self_batch, other_batch, axis=0)
    )
    self.assertDataclassAlmostEqual(expected_state, self_state)

  @parameterized.named_parameters([
      dict(
          testcase_name='init_state',
          add_batch=False,
      ),
      dict(
          testcase_name='non_empty_1d_batch',
          add_batch=True,
      ),
      dict(
          testcase_name='non_empty_multi_dim_batch',
          add_batch=True,
          multi_dimension=True,
      ),
  ])
  def test_mean_and_variance_properties(self, add_batch, multi_dimension=False):
    batch = np.array([1, 2, 3, np.nan] * int(add_batch))
    if multi_dimension:
      batch = np.tile(batch, (2, 1))
    state = rolling_stats.MeanAndVariance()
    if add_batch:
      state.add(batch)

    expected_properties_dict = {
        'mean': np.nanmean(batch, axis=0),
        'var': np.nanvar(batch, axis=0),
        'stddev': np.nanstd(batch, axis=0),
        'count': np.nansum(~np.isnan(batch), axis=0),
        'total': np.nansum(batch, axis=0),
    }
    for property_name, value in expected_properties_dict.items():
      np.testing.assert_allclose(getattr(state.result(), property_name), value)

  def test_mean_and_variance_agg_fn(self):
    agg_fn = rolling_stats.MeanAndVariance().as_agg_fn()
    batches = np.arange(3)
    actual = agg_fn(batches)
    self.assertDataclassAlmostEqual(
        get_expected_mean_and_variance(batches), actual
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='1d_batch',
          batch=[1, 2],
          expected_str=(
              'count: 2, total: 3.0, mean: 1.5, var: 0.25, stddev: 0.5'
          ),
      ),
      dict(
          testcase_name='multi_dim_batch',
          batch=[[1, 2], [2, 4]],
          expected_str=(
              'count: [2 2], total: [3. 6.], mean: [1.5 3. ], var: [0.25 1.  ],'
              ' stddev: [0.5 1. ]'
          ),
      ),
      dict(
          testcase_name='empty_batch',
          batch=[],
          expected_str='count: 0, total: 0, mean: nan, var: nan, stddev: nan',
      ),
  ])
  def test_mean_and_variance_string_representation(self, batch, expected_str):
    state = rolling_stats.MeanAndVariance()
    state.add(batch)
    self.assertEqual(str(state), expected_str)

  @parameterized.named_parameters([
      dict(
          testcase_name='empty_input',
          inputs={},
          expected={},
      ),
      dict(
          testcase_name='empty_list',
          inputs=[],
          expected=[],
      ),
      dict(
          testcase_name='empty_tuple',
          inputs=(),
          expected=(),
      ),
      dict(
          testcase_name='dict_input',
          inputs={'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])},
          expected={
              'a': rolling_stats.MeanAndVariance()([1, 2, 3]),
              'b': rolling_stats.MeanAndVariance()([4, 5, 6]),
          },
      ),
      dict(
          testcase_name='dict_tuple_input',
          inputs={
              'a': (np.array([1, 2, 3]), np.array([4, 5, 6])),
              'b': np.array([7, 8, 9]),
          },
          expected={
              'a': (
                  rolling_stats.MeanAndVariance()([1, 2, 3]),
                  rolling_stats.MeanAndVariance()([4, 5, 6]),
              ),
              'b': rolling_stats.MeanAndVariance()([7, 8, 9]),
          },
      ),
      dict(
          testcase_name='list_input',
          inputs=[np.array([1, 2, 3]), np.array([4, 5, 6])],
          expected=[
              rolling_stats.MeanAndVariance()([1, 2, 3]),
              rolling_stats.MeanAndVariance()([4, 5, 6]),
          ],
      ),
      dict(
          testcase_name='tuple_input',
          inputs=(np.array([1, 2, 3]), np.array([4, 5, 6])),
          expected=(
              rolling_stats.MeanAndVariance()([1, 2, 3]),
              rolling_stats.MeanAndVariance()([4, 5, 6]),
          ),
      ),
      dict(
          testcase_name='batch_score_input',
          inputs=np.array([[1, 2, 3], [4, 5, 6]]),
          batch_score_fn=lambda x: {'x': x[0], 'y': x[1]},
          expected={
              'x': rolling_stats.MeanAndVariance()([1, 2, 3]),
              'y': rolling_stats.MeanAndVariance()([4, 5, 6]),
          },
      ),
  ])
  def test_nested_agg_fn_nested(self, inputs, expected, batch_score_fn=None):
    agg_fn = rolling_stats.MeanAndVariance(
        batch_score_fn=batch_score_fn
    ).as_agg_fn(nested=True)
    actual = agg_fn(inputs)
    test_utils.assert_nested_container_equal(
        self, expected, actual, strict=True
    )


class MinMaxAndCountTest(parameterized.TestCase):
  def test_min_max_and_count_merge(self):
    batch_1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)  # len(batch_1) = 9
    batch_2 = (8, 6, 7, 5, 3, 0, 9)  # len(batch_2) = 7
    batch_3 = (5, 4, 3, 2, 1)  # len(batch_3) = 5

    expected_result = rolling_stats.MinMaxAndCount(
        _count=21,  # len(batch_1) + len(batch_2) + len(batch_3)
        _min=5,  # min(len(batch_1), len(batch_2), len(batch_3))
        _max=9,  # max(len(batch_1), len(batch_2), len(batch_3))
    )

    state_1 = rolling_stats.MinMaxAndCount(batch_score_fn=len).add(batch_1)
    state_2 = rolling_stats.MinMaxAndCount(batch_score_fn=len).add(batch_2)
    state_3 = rolling_stats.MinMaxAndCount(batch_score_fn=len).add(batch_3)

    self.assertEqual(state_1.merge(state_2).merge(state_3), expected_result)

  def test_min_max_and_count_len(self):
    batch_1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)  # len(batch_1) = 9
    batch_2 = (8, 6, 7, 5, 3, 0, 9)  # len(batch_2) = 7
    batch_3 = (5, 4, 3, 2, 1)  # len(batch_3) = 5

    expected_properties_dict = {
        'count': 21,  # len(batch_1) + len(batch_2) + len(batch_3)
        'min': 5,  # min(len(batch_1), len(batch_2), len(batch_3))
        'max': 9,  # max(len(batch_1), len(batch_2), len(batch_3))
    }

    state = rolling_stats.MinMaxAndCount(batch_score_fn=len)
    for batch in (batch_1, batch_2, batch_3):
      state.add(batch)

    for property_name, value in expected_properties_dict.items():
      self.assertEqual(getattr(state, property_name), value)

  def test_min_max_and_count_custom_batch_score_fn(self):
    batch_1 = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (2, 4, 6, 8, 10, 12, 14, 16, 18))
    batch_2 = ((1, 2, 3, 4, 5, 6, 7), (8, 6, 7, 5, 3, 0, 9))
    batch_3 = ((1, 2, 3, 4, 5), (5, 4, 3, 2, 1))

    expected_properties_dict = {
        'count': 42,  # 2 * 9 + 2 * 7 + 2 * 5 = 42
        'min': 10,  # min(2 * 9, 2 * 7, 2 * 5) = 2 * 5 = 10
        'max': 18,  # max(2 * 9, 2 * 7, 2 * 5) = 2 * 9 = 18
    }

    num_elem = lambda input: sum([len(batch) for batch in input])
    state = rolling_stats.MinMaxAndCount(batch_score_fn=num_elem)
    for batch in (batch_1, batch_2, batch_3):
      state.add(batch)

    for property_name, value in expected_properties_dict.items():
      self.assertEqual(getattr(state, property_name), value)

  @parameterized.named_parameters(
      dict(
          testcase_name='mo_batch_score_fn',
          batch_score_fn=None,
          expected_min=0,  # min(batch_1, batch_2, batch_3) = 0
          expected_max=18,  # max(batch_1, batch_2, batch_3) = 18
      ),
      dict(
          testcase_name='np_sum',
          batch_score_fn=np.sum,
          # min(sum(batch_1), sum(batch_2), sum(batch_3)) = min(135, 38, 30)
          expected_min=30,
          # max(sum(batch_1), sum(batch_2), sum(batch_3)) = max(135, 38, 30)
          expected_max=135,
      ),
  )
  def test_min_max_and_count_mixed_dim_inputs_np_sum(
      self, batch_score_fn, expected_min, expected_max
  ):
    batch_1 = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (2, 4, 6, 8, 10, 12, 14, 16, 18))
    batch_2 = (8, 6, 7, 5, 3, 0, 9)
    batch_3 = ((1, 2, 3, 4, 5), (5, 4, 3, 2, 1))

    expected_properties_dict = {
        'count': 35,  # 2 * 9 + 7 + 2 * 5 = 35
        'min': expected_min,
        'max': expected_max,
    }

    state = rolling_stats.MinMaxAndCount(batch_score_fn=batch_score_fn)
    for batch in (batch_1, batch_2, batch_3):
      state.add(batch)

    for property_name, value in expected_properties_dict.items():
      self.assertEqual(getattr(state, property_name), value)

  @parameterized.named_parameters(
      dict(
          testcase_name='axis_none',
          batch_score_fn=None,
          axis=None,
          expected_min=0,  # min of all elements
          expected_max=18,  # max of all elements
      ),
      dict(
          testcase_name='axis_0',
          batch_score_fn=None,
          axis=0,
          # np.minimum.reduce((
          # (1, 2, 3, 4, 5, 6, 7, 8, 9),
          # (1, 2, 3, 4, 3, 0, 7, 0, 0),
          # (1, 2, 3, 4, 4, 4, 3, 2, 1),
          # ))
          # = [1 2 3 4 3 0 3 0 0]
          expected_min=(1, 2, 3, 4, 3, 0, 3, 0, 0),
          # np.maximum.reduce((
          # (2, 4, 6, 8, 10, 12, 14, 16, 18),
          # (8, 6, 7, 5, 5, 6, 9, 9, 9),
          # (4, 4, 4, 4, 5, 4, 4, 4, 4),
          # ))
          # = [ 8  6  7  8 10 12 14 16 18]
          expected_max=(8, 6, 7, 8, 10, 12, 14, 16, 18),
      ),
      dict(
          testcase_name='axis_1',
          batch_score_fn=None,
          axis=1,
          # np.minimum.reduce(((1, 2), (0, 0), (1, 4))) = [0 0]
          expected_min=(0, 0),
          # np.maximum.reduce(((9, 18), (7, 9), (5, 4))) = [ 9 18]
          expected_max=(9, 18),
      ),
  )
  def test_min_max_and_count_axis(
      self, batch_score_fn, axis, expected_min, expected_max
  ):
    batch_1 = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (2, 4, 6, 8, 10, 12, 14, 16, 18))
    batch_2 = ((1, 2, 3, 4, 5, 6, 7, 0, 0), (8, 6, 7, 5, 3, 0, 9, 9, 9))
    batch_3 = ((1, 2, 3, 4, 5, 4, 3, 2, 1), (4, 4, 4, 4, 4, 4, 4, 4, 4))

    state = rolling_stats.MinMaxAndCount(
        batch_score_fn=batch_score_fn, axis=axis
    )
    for batch in (batch_1, batch_2, batch_3):
      state.add(batch)

    self.assertEqual(state.count, 54)  # 3 * 2 * 9
    np.testing.assert_array_equal(state.min, expected_min)
    np.testing.assert_array_equal(state.max, expected_max)

  def test_min_max_and_count_one_large_batch(self):
    num_inputs = 1000000

    inputs = np.random.random_sample(size=num_inputs)

    expected_properties = ('count', 'min', 'max')

    minmax = rolling_stats.MinMaxAndCount(batch_score_fn=len).as_agg_fn()
    actual_result = minmax(inputs)

    for property_name in expected_properties:
      self.assertEqual(getattr(actual_result, property_name), num_inputs)

  def test_min_max_and_count_many_batches(self):
    num_batches = 1000
    batch_size = 10000
    inputs = np.random.random_sample(size=(num_batches, batch_size))

    expected_properties_dict = {
        'count': num_batches * batch_size,
        'min': batch_size,
        'max': batch_size,
    }

    state = rolling_stats.MinMaxAndCount(batch_score_fn=len)
    for input_batch in inputs:
      state.add(input_batch)

    for property_name, value in expected_properties_dict.items():
      self.assertEqual(getattr(state, property_name), value)

  def test_min_max_and_count_empty_input(self):
    empty_batch = ()

    expected_properties = ('count', 'min', 'max')

    actual_result = rolling_stats.MinMaxAndCount(batch_score_fn=len).add(
        empty_batch
    )

    for property_name in expected_properties:
      # All properties should be 0 for an empty batch.
      self.assertEqual(getattr(actual_result, property_name), 0)


class ValueAccumulatorTest(parameterized.TestCase):

  def test_value_accumulator_single_column_no_concatenate(self):
    accumulator = rolling_stats.ValueAccumulator()
    inputs = [1, 2, 3]
    for batch in inputs:
      accumulator.add(batch)
    actual = accumulator.result()
    self.assertEqual(inputs, actual)

  def test_value_accumulator_concat_list(self):
    concat_fn = lambda x, y: x + y
    accumulator = rolling_stats.ValueAccumulator(concat_fn=concat_fn)
    inputs = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for batch in inputs:
      accumulator.add(batch)
    actual = accumulator.result()
    self.assertEqual(list(range(9)), actual)

  def test_value_accumulator_concat_np_array(self):
    concat_fn = lambda x, y: np.concat((x, y), axis=-1)
    accumulator = rolling_stats.ValueAccumulator(concat_fn=concat_fn)
    inputs = [np.arange(3), np.arange(3, 6), np.arange(6, 9)]
    for batch in inputs:
      accumulator.add(batch)
    actual = accumulator.result()
    np.testing.assert_array_equal(actual, np.arange(9))

  def test_value_accumulator_two_columns(self):
    concat_fn = lambda x, y: x + y
    accumulator = rolling_stats.ValueAccumulator(concat_fn=concat_fn)
    inputs = [([0, 1, 2], [3, 4, 5]), ([6, 7, 8], [9, 10, 11])]
    for batch in inputs:
      accumulator.add(*batch)
    actual = accumulator.result()
    expected = ([0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11])
    self.assertEqual(expected, actual)

  def test_value_accumulator_metric_fns(self):
    concat_fn = lambda x, y: x + y
    metric_fns = {'sum': sum, 'mean': np.mean}
    accumulator = rolling_stats.ValueAccumulator(concat_fn, metric_fns)
    inputs = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for batch in inputs:
      accumulator.add(batch)
    actual = accumulator.result()
    expected = {'sum': 36, 'mean': 4.0}
    self.assertEqual(expected, actual)

  def test_value_accumulator_as_agg_fn(self):
    concat_fn = lambda x, y: x + y
    accumulator = rolling_stats.ValueAccumulator(concat_fn, sum)
    agg_fn = accumulator.as_agg_fn()
    actual = agg_fn(list(range(9)))
    expected = 36
    self.assertEqual(expected, actual)


class R2TjurTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='absolute', r2_metric=rolling_stats.R2Tjur),
      dict(testcase_name='relative', r2_metric=rolling_stats.R2TjurRelative),
  )
  def test_r2_tjur_merge(self, r2_metric):
    y_true_1 = (0, 1)
    y_pred_1 = (0.8, 0.3)

    y_true_2 = 1
    y_pred_2 = 0.9

    state_1 = r2_metric().add(y_true_1, y_pred_1)
    state_2 = r2_metric().add(y_true_2, y_pred_2)
    result = state_1.merge(state_2)

    # sum(y_true) = 0.0 + 1.0 + 1.0  = 2.0
    # sum(y_true * y_pred) = 0.0 * 0.8 + 1.0 * 0.3 + 1.0 * 0.9 = 1.2
    # sum(1 - y_true) = 1.0 + 0.0 + 0.0 = 1.0
    # sum((1 - y_true) * y_pred) = 1.0 * 0.8 + 0.0 * 0.3 + 0.0 * 0.9 = 0.8

    expected_result = r2_metric(2.0, 1.2, 1.0, 0.8)

    self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      # sum_y_true = 2.0, sum_y_pred = 1.2
      # sum_neg_y_true = 1.0, sum_neg_y_pred = 0.1
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=(1.2 / 2.0) - (0.8 / 1.0),
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=(1.2 / 2.0) / (0.8 / 1.0),
      ),
  )
  def test_r2_tjur_simple(self, r2_metric, expected_result):
    y_true = (0, 1, 1)
    y_pred = (0.8, 0.3, 0.9)

    actual_result = r2_metric().add(y_true, y_pred).result()

    self.assertAlmostEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=-9.78031226162579e-05,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=0.9998043715885978,
      ),
  )
  def test_r2_tjur_one_large_batch(self, r2_metric, expected_result):
    np.random.seed(seed=0)

    y_true = np.random.rand(1000000)
    y_pred = np.random.rand(1000000)

    actual_result = r2_metric().as_agg_fn()(y_true, y_pred)

    self.assertAlmostEqual(actual_result, expected_result, places=14)

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=-2340.933593683032,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=1.0129655857337465,
      ),
  )
  def test_r2_tjur_many_batches_little_correlation(
      self, r2_metric, expected_result
  ):
    np.random.seed(seed=0)

    y_true = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))
    y_pred = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = r2_metric()
    for y_true_i, y_pred_i in zip(y_true, y_pred):
      state.add(y_true_i, y_pred_i)

    self.assertAlmostEqual(state.result(), expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=0.9999901799905766,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=214595.2102587055,
      ),
  )
  def test_r2_tjur_many_batches_much_correlation(
      self, r2_metric, expected_result
  ):
    np.random.seed(seed=0)

    y_true = np.random.uniform(low=1e-5, high=1 - 1e-5, size=(10000, 10000))

    # This is a noisy version of y_true.
    y_pred = y_true + np.random.uniform(
        low=-1e-5, high=1e-5, size=(10000, 10000)
    )

    y_true = np.round(y_true)
    y_pred = np.round(y_pred)

    state = r2_metric()
    for y_true_i, y_pred_i in zip(y_true, y_pred):
      state.add(y_true_i, y_pred_i)

    self.assertAlmostEqual(state.result(), expected_result, places=10)

  def test_r2_tjur_absolute_many_batches_direct_correlation(self):
    y = np.round(np.random.uniform(size=(10000, 10000)))

    state = rolling_stats.R2Tjur()
    for y_i in y:
      state.add(y_i, y_i)

    expected_result = 1

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  def test_r2_tjur_relative_many_batches_direct_correlation(self):
    y = np.round(np.random.uniform(size=(10000, 10000)))

    state = rolling_stats.R2TjurRelative()
    for y_i in y:
      state.add(y_i, y_i)

    self.assertTrue(math.isnan(state.result()))

  @parameterized.named_parameters(
      dict(
          testcase_name='absolute',
          r2_metric=rolling_stats.R2Tjur,
          expected_result=-1,
      ),
      dict(
          testcase_name='relative',
          r2_metric=rolling_stats.R2TjurRelative,
          expected_result=0,
      ),
  )
  def test_r2_tjur_many_batches_inverse_correlation(
      self, r2_metric, expected_result
  ):
    y = np.round(np.random.uniform(size=(10000, 10000)))

    state = r2_metric()
    for y_i in y:
      state.add(y_i, 1 - y_i)

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  @parameterized.named_parameters(
      dict(testcase_name='y_true_is_0', y_true=(0, 0, 0)),
      dict(testcase_name='y_true_is_1', y_true=(1, 1, 1)),
  )
  def test_r2_tjur_absolute_returns_nan(self, y_true):
    y_pred = (1, 0, 1)

    self.assertTrue(
        math.isnan(rolling_stats.R2Tjur().add(y_true, y_pred).result())
    )

  @parameterized.named_parameters(
      dict(testcase_name='y_true_is_0', y_true=(0, 0, 0), y_pred=(1, 0, 1)),
      dict(
          testcase_name='sum_neg_y_pred_is_0',
          # all(y_t == 1 or y_p == 0 for y_t, y_p in zip(y_true, y_pred))
          y_true=(1, 0, 1),
          y_pred=(1, 0, 1),
      ),
  )
  def test_r2_tjur_relative_returns_nan(self, y_true, y_pred):
    self.assertTrue(
        math.isnan(rolling_stats.R2TjurRelative().add(y_true, y_pred).result())
    )


class PartialCrossFeatureStatsTest(absltest.TestCase):

  def test_partial_cross_feature_stats_merge(self):
    x_1 = (1, 2, 3, 4)
    y_1 = (10, 9, 2.5, 6)

    x_2 = (5, 6, 7)
    y_2 = (4, 3, 2)

    state_1 = rolling_stats.RRegression().add(x_1, y_1)
    state_2 = rolling_stats.RRegression().add(x_2, y_2)
    result = state_1.merge(state_2)

    expected_result = rolling_stats.RRegression(
        num_samples=7,  # len(x_1) + len(x_2)
        sum_x=28,  # 1 + 2 + 3 + 4 + 5 + 6 + 7
        sum_y=36.5,  # 10 + 9 + 2.5 + 6 + 4 + 3 + 2
        sum_xx=140,  # 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2
        sum_yy=252.25,  # 10^2 + 9^2 + 2.5^2 + 6^2 + 4^2 + 3^2 + 2^2
        sum_xy=111.5,  # 1*10 + 2*9 + 3*2.5 + 4*6 + 5*4 + 6*3 + 7*2
    )

    self.assertEqual(result, expected_result)


class CovarianceTest(absltest.TestCase):

  def test_covariance_single_output(self):
    x = (1, 2, 3, 4, 5, 6, 7)
    y = (10, 9, 2.5, 6, 4, 3, 2)

    # covariance(X, Y) = [sum(XY) - sum(X) * sum(Y) / num_samples] / num_samples
    # = (111.5 - 28 * 36.5 / 7) / 7 = -4.928571428571429
    expected_result = -4.928571428571429

    actual_result = rolling_stats.Covariance().add(x, y).result()

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  def test_covariance_single_output_as_agg_fn(self):
    x = (1, 2, 3, 4, 5, 6, 7)
    y = (10, 9, 2.5, 6, 4, 3, 2)

    # covariance(X, Y) = [sum(XY) - sum(X) * sum(Y) / num_samples] / num_samples
    # = (111.5 - 28 * 36.5 / 7) / 7 = -4.928571428571429
    expected_result = -4.928571428571429

    actual_result = rolling_stats.Covariance().as_agg_fn()(x, y)

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  def test_covariance_multi_output(self):
    x1 = (10, 9, 2.5, 6, 4, 3, 2)
    x2 = (8, 6, 7, 5, 3, 0, 9)
    y = (1, 2, 3, 4, 5, 6, 7)

    # covariance(X, Y) = [sum(XY) - sum(X) * sum(Y) / num_samples] / num_samples
    # covariance(x1, y) = (111.5 - 36.5 * 28 / 7) / 7 = -4.928571428571429
    # covariance(x2, y) = (139 - 38 * 28 / 7) / 7 = -1.8571428571428572
    expected_result = (-4.928571428571429, -1.8571428571428572)

    x_all = np.array((x1, x2)).T

    actual_result = rolling_stats.Covariance().add(x_all, y).result()

    np.testing.assert_almost_equal(actual_result, expected_result)

  def test_covariance_multi_output_as_agg_fn(self):
    x1 = (10, 9, 2.5, 6, 4, 3, 2)
    x2 = (8, 6, 7, 5, 3, 0, 9)
    y = (1, 2, 3, 4, 5, 6, 7)

    # covariance(X, Y) = [sum(XY) - sum(X) * sum(Y) / num_samples] / num_samples
    # covariance(x1, y) = (111.5 - 36.5 * 28 / 7) / 7 = -4.928571428571429
    # covariance(x2, y) = (139 - 38 * 28 / 7) / 7 = -1.8571428571428572
    expected_result = (-4.928571428571429, -1.8571428571428572)

    x_all = np.array((x1, x2)).T

    actual_result = rolling_stats.Covariance().as_agg_fn()(x_all, y)

    np.testing.assert_almost_equal(actual_result, expected_result)


class RRegressionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='centered',
          center=True,
          # From
          # sklearn.feature_selection.r_regression(
          # X=np.reshape(x, (-1, 1)), y=y
          # )[0]
          expected_result=-0.8285038835884279,
      ),
      dict(
          testcase_name='not_centered',
          center=False,
          # From
          # sklearn.feature_selection.r_regression(
          # X=np.reshape(x, (-1, 1)), y=y, center=False
          # )[0]
          expected_result=0.5933285714624903,
      ),
  )
  def test_r_regression_single_output(self, center, expected_result):
    x = (1, 2, 3, 4, 5, 6, 7)
    y = (10, 9, 2.5, 6, 4, 3, 2)

    actual_result = rolling_stats.RRegression(center=center).as_agg_fn()(x, y)

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  @parameterized.named_parameters(
      dict(
          testcase_name='centered',
          center=True,
          # From sklearn.feature_selection.r_regression(x_all, y)
          expected_result=(-0.82850388, -0.32338709),
      ),
      dict(
          testcase_name='not_centered',
          center=False,
          # From
          # sklearn.feature_selection.r_regression(X=x_all, y=y, center=False)
          expected_result=(0.59332857, 0.72301752),
      ),
  )
  def test_r_regression_multi_output(self, center, expected_result):
    x1 = (10, 9, 2.5, 6, 4, 3, 2)
    x2 = (8, 6, 7, 5, 3, 0, 9)
    y = (1, 2, 3, 4, 5, 6, 7)

    x_all = np.array((x1, x2)).T

    actual_result = (
        rolling_stats.RRegression(center=center).add(x_all, y).result()
    )

    np.testing.assert_almost_equal(actual_result, expected_result)

  def test_r_regression_one_large_batch(self):
    np.random.seed(seed=0)

    x = np.random.rand(1000000)
    y = np.random.rand(1000000)

    actual_result = rolling_stats.RRegression().add(x, y).result()

    # From
    # sklearn.feature_selection.r_regression(X=np.reshape(x, (-1, 1)), y=y)[0]
    expected_result = -0.0002932187695762664

    self.assertAlmostEqual(actual_result, expected_result, places=13)

  def test_r_regression_many_batches_little_correlation(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))
    y = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    # From
    # sklearn.feature_selection.r_regression(X=np.reshape(x, (-1, 1)), y=y)[0]
    expected_result = 4.231252166807617e-05

    self.assertAlmostEqual(state.result(), expected_result, places=14)

  def test_r_regression_many_batches_much_correlation(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    # This is a noisy version of x.
    y = x + np.random.uniform(low=-1e5, high=1e5, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    # From
    # sklearn.feature_selection.r_regression(X=np.reshape(x, (-1, 1)), y=y)[0]
    expected_result = 0.995037725730923

    self.assertAlmostEqual(state.result(), expected_result, places=10)

  def test_r_regression_many_batches_direct_correlation(self):
    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i in x:
      state.add(x_i, x_i)

    expected_result = 1

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  def test_r_regression_many_batches_inverse_correlation(self):
    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    state = rolling_stats.RRegression()
    for x_i in x:
      state.add(x_i, -x_i)

    expected_result = -1

    self.assertAlmostEqual(state.result(), expected_result, places=9)

  @parameterized.named_parameters(
      dict(testcase_name='empty_input', x=(), y=()),
      dict(testcase_name='0_input', x=(0, 0), y=(0, 0)),
  )
  def test_r_regression_returns_nan(self, x, y):
    self.assertTrue(math.isnan(rolling_stats.RRegression().add(x, y).result()))

  def test_r_regression_valid_and_0_input(self):
    x_valid = (10, 9, 2.5, 6, 4, 3, 2)
    x_0 = (0, 0, 0, 0, 0, 0, 0)
    y = (1, 2, 3, 4, 5, 6, 7)

    x_all = np.array((x_valid, x_0)).T

    actual_result = rolling_stats.RRegression().add(x_all, y).result()

    # From sklearn.feature_selection.r_regression(x_all, y)
    expected_result = (-0.82850388, float('nan'))

    np.testing.assert_almost_equal(actual_result, expected_result)


class SymmetricPredictionDifferenceTest(absltest.TestCase):

  def test_symmetric_prediction_difference_merge(self):
    x_1 = (0, 1)
    y_1 = (0.8, 0.3)

    x_2 = 1
    y_2 = 0.9

    state_1 = rolling_stats.SymmetricPredictionDifference().add(x_1, y_1)
    state_2 = rolling_stats.SymmetricPredictionDifference().add(x_2, y_2)
    result = state_1.merge(state_2)

    # sum_half_pointwise_rel_diff =
    # (|0 - 0.8| / |0 + 0.8|
    # + |1 - 0.3| / |1 + 0.3|
    # + |1 - 0.9| / |1 + 0.9|)
    # = 1.59109311741

    self.assertEqual(result.num_samples, 3)
    self.assertAlmostEqual(
        result.sum_half_pointwise_rel_diff, 1.59109311741, places=11
    )

  def test_symmetric_prediction_difference_simple(self):
    x = (0, 1, 1)
    y = (0.8, 0.3, 0.9)

    expected_result = 1.06072874494  # 2 * 1.59109311741 / 3 = 1.06072874494

    spd = rolling_stats.SymmetricPredictionDifference().as_agg_fn()
    actual_result = spd(x, y)

    self.assertAlmostEqual(actual_result, expected_result, places=11)

  def test_symmetric_prediction_difference_one_large_batch(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=1000000)
    y = np.random.uniform(low=-1e6, high=1e6, size=1000000)

    actual_result = (
        rolling_stats.SymmetricPredictionDifference().add(x, y).result()
    )

    expected_result = 32.611545081600894

    self.assertAlmostEqual(actual_result, expected_result, places=10)

  def test_symmetric_prediction_difference_many_batches_little_correlation(
      self,
  ):
    np.random.seed(seed=0)

    x = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))
    y = np.random.uniform(low=-1e6, high=1e6, size=(10000, 10000))

    expected_result = 36.23844148369455

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    self.assertAlmostEqual(state.result(), expected_result, places=10)

  def test_symmetric_prediction_difference_many_batches_much_correlation(self):
    np.random.seed(seed=0)

    x = np.random.uniform(low=1e-5, high=1 - 1e-5, size=(10000, 10000))

    # y is a noisy version of x.
    y = x + np.random.uniform(low=-1e-5, high=1e-5, size=(10000, 10000))

    expected_result = 5.8079104443249064e-05

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i, y_i in zip(x, y):
      state.add(x_i, y_i)

    self.assertAlmostEqual(state.result(), expected_result, places=16)

  def test_symmetric_prediction_difference_many_identical_batches(self):
    x = np.random.uniform(size=(10000, 10000))

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i in x:
      # sum_half_pointwise_rel_diff should be 0 for every point.
      state.add(x_i, x_i)

    expected_result = 0

    self.assertAlmostEqual(state.result(), expected_result, places=11)

  def test_symmetric_prediction_difference_many_batches_opposite(self):
    x = np.random.uniform(size=(10000, 10000))

    state = rolling_stats.SymmetricPredictionDifference()
    for x_i in x:
      # sum_half_pointwise_rel_diff should remain 0 because all of the pointwise
      # average relative differences are undefined.
      state.add(x_i, -x_i)

    expected_result = 0

    self.assertAlmostEqual(state.result(), expected_result, places=11)

  def test_symmetric_prediction_difference_absolute_returns_nan(self):
    x_empty = ()

    self.assertTrue(
        math.isnan(
            rolling_stats.SymmetricPredictionDifference()
            .add(x_empty, x_empty)
            .result()
        )
    )

  def test_symmetric_prediction_difference_asserts_with_invalid_input(self):
    # x.shape != y.shape
    x = (1, 2, 3)
    y = (4, 5)

    expected_error_message = (
        r'SymmetricPredictionDifference\.add\(\) requires x and y to have the'
        r' same shape, but recieved x=\[1\. 2\. 3\.\] and y=\[4\. 5\.\] with'
        r' x.shape=\(3\,\) and y.shape=\(2\,\)'
    )

    metric = rolling_stats.SymmetricPredictionDifference()

    with self.assertRaisesRegex(ValueError, expected_error_message):
      metric.add(x, y)


if __name__ == '__main__':
  absltest.main()
