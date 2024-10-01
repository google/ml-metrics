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

    actual_result = (
        rolling_stats.Histogram(range=bin_range, bins=bins)
        .add(input_1)
        .add(input_2)
        .result()
    )

    expected_histogram = (
        # Input values in each bin:
        3,  # (0, 0, 0)
        2,  # (0.2, 0.2)
        2,  # (0.5, 0.5)
        0,  # (),
        7,  # (0.8, 0.8, 1, 1, 1, 1, 1)
    )
    expected_bin_edges = (0, 0.2, 0.4, 0.6, 0.8, 1)

    np.testing.assert_equal(actual_result.hist, expected_histogram)
    np.testing.assert_allclose(
        actual_result.bin_edges, expected_bin_edges
    )

  def test_histogram_one_large_batch(self):
    np.random.seed(seed=0)

    num_values = 1000000
    bins = 10
    left_boundary = -1e6
    right_boundary = 1e6

    inputs = np.random.uniform(
        low=left_boundary, high=right_boundary, size=num_values
    )

    actual_result = (
        rolling_stats.Histogram(
            range=(left_boundary, right_boundary),
            bins=bins,
        )
        .add(inputs)
        .result()
    )

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

    hist_1 = rolling_stats.Histogram(place_holder_range)
    hist_1._hist = place_holder_hist
    hist_1._bin_edges = np.array(bin_edges_1)

    with self.assertRaisesRegex(
        ValueError,
        'The bin edges of the two Histograms must be equal, but recieved'
    ):
      hist_1._merge(place_holder_hist, np.array(bin_edges_2))


def get_expected_mean_and_variance(batches, batch_score_fn=None):
  if batch_score_fn is not None:
    batches = [batch_score_fn(batch) for batch in batches]
  batches = np.asarray(batches)
  batch = np.asarray(batches).reshape(-1, *batches.shape[2:])
  return rolling_stats.MeanAndVariance(
      batch_score_fn=batch_score_fn,
      _mean=np.nanmean(batch, axis=0),
      _var=np.nanvar(batch, axis=0),
      _count=np.nansum(~np.isnan(batch), axis=0),
  )


class FixedSizeSampleTest(parameterized.TestCase):
  def _assert_fixed_size_samples_equal(self, actual, expected):
    self.assertEqual(actual.max_size, expected.max_size)
    self.assertEqual(actual._random_seed, expected._random_seed)
    self.assertSameElements(actual._reservoir, expected._reservoir)
    self.assertEqual(
        actual._num_samples_reviewed, expected._num_samples_reviewed
    )

  def _batch_data(self, data, batch_size):
    """Splits data into batches of equal length."""
    for i in range(0, len(data), batch_size):
      yield data[i : i + batch_size]

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
        _random_seed=0,
        _reservoir=reservoir_original,
        _num_samples_reviewed=num_samples_original,
    )
    other = rolling_stats.FixedSizeSample(
        max_size=5,
        _reservoir=reservoir_other,
        _num_samples_reviewed=num_samples_other,
    )

    expected_result = rolling_stats.FixedSizeSample(
        max_size=5,
        _random_seed=0,
        _reservoir=expected_reservoir,
        _num_samples_reviewed=expected_num_samples_reviewed,
    )

    self._assert_fixed_size_samples_equal(
        original.merge(other), expected_result
    )

  def test_fixed_size_sample_merge_different_max_sizes(self):
    max_size_1 = 5
    max_size_2 = 10

    res_1 = [1, 2, 3, 4, 5]
    res_2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    original = rolling_stats.FixedSizeSample(
        max_size=max_size_1,
        _random_seed=0,
        _reservoir=res_1,
        _num_samples_reviewed=10,
    )
    other = rolling_stats.FixedSizeSample(
        max_size=max_size_2, _reservoir=res_2, _num_samples_reviewed=10
    )

    original.merge(other)

    expected_result = (2, 1, 70, 100, 60)

    np.testing.assert_array_equal(original.result(), expected_result)

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

    actual_reservoir = (
        rolling_stats.FixedSizeSample(max_size=8, _random_seed=random_seed)
        .add(stream)
        .result()
    )

    np.testing.assert_array_equal(actual_reservoir, expected_reservoir)

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

    actual_reservoir = (
        rolling_stats.FixedSizeSample(max_size=4, _random_seed=0)
        .add(stream)
        .result()
    )

    expected_reservoir = ('Fig', 'Elderberry', 'Jackfruit', 'Honeydew')

    np.testing.assert_array_equal(actual_reservoir, expected_reservoir)

  def test_fixed_size_sample_one_large_batch(self):
    batch_size = 1000000
    datastream = np.random.default_rng(0).uniform(
        low=-1e6, high=1e6, size=batch_size
    )

    actual_reservoir = (
        rolling_stats.FixedSizeSample(max_size=10, _random_seed=0)
        .add(datastream)
        .result()
    )

    expected_reservoir = (
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

    self.assertSequenceAlmostEqual(
        actual_reservoir, expected_reservoir, places=9
    )

  def test_fixed_size_sample_many_batches(self):
    size = (1000, 1000)
    datastreams = np.random.default_rng(0).uniform(
        low=-1e6, high=1e6, size=size
    )

    metric = rolling_stats.FixedSizeSample(max_size=10, _random_seed=0)
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

  def test_fixed_size_sample_algorithm(self):
    # This tests checks that the algorithm for generating the reservoir is
    # correct.
    seeds = 2  # Number of seeds to test.
    num_reservoirs = 300  # Number of times to run the algorithm.
    batches = 50  # Number of times to add a batch of data.
    max_sample = 25  # Max of the samples.

    res_len = 10  # Max length of the reservoir.
    batch_len = 100  # Number of samples per batch.

    margin_of_error = 0.25

    final_num_samples = num_reservoirs * res_len
    ideal_final_count = final_num_samples / max_sample
    min_final_count = ideal_final_count * (1 - margin_of_error)
    max_final_count = ideal_final_count * (1 + margin_of_error)

    for seed in range(seeds):
      rng = np.random.default_rng(seed)
      sample_histo = [0] * max_sample

      for _ in range(num_reservoirs):
        # Generate a reservoir.
        sampler = rolling_stats.FixedSizeSample(
            max_size=res_len, _random_seed=0
        )
        for _ in range(batches):
          sampler.add(
              rng.integers(
                  low=0, high=max_sample, size=batch_len, endpoint=False
              )
          )

        # Update the histogram with the reservoir values.
        for sample in sampler.result():
          sample_histo[sample] += 1

      # Assert that all the reservoir samples were added to the histogram.
      self.assertEqual(sum(sample_histo), final_num_samples)

      # Assert that the histogram values are within the margin of error.
      self.assertTrue(all(bucket > min_final_count for bucket in sample_histo))
      self.assertTrue(all(bucket < max_final_count for bucket in sample_histo))

  def test_fixed_size_sample_small_batch_size(self):
    num_seeds = 50000
    samples_per_seed = 30
    batch_size = 3
    res_len = 10
    margin_of_error = 0.02

    final_num_samples = num_seeds * res_len
    ideal_final_count = final_num_samples / samples_per_seed
    min_final_count = ideal_final_count * (1 - margin_of_error)
    max_final_count = ideal_final_count * (1 + margin_of_error)

    sample_histo = [0] * samples_per_seed
    for seed in range(num_seeds):
      sampler = rolling_stats.FixedSizeSample(
          max_size=res_len, _random_seed=seed
      )

      for batch in self._batch_data(range(samples_per_seed), batch_size):
        sampler.add(batch)

      for sample in sampler.result():
        sample_histo[int(sample)] += 1

    # Assert that all the reservoir samples were added to the histogram.
    self.assertEqual(sum(sample_histo), final_num_samples)

    # Assert that the histogram values are within the margin of error.
    self.assertTrue(all(bucket > min_final_count for bucket in sample_histo))
    self.assertTrue(all(bucket < max_final_count for bucket in sample_histo))

  def test_fixed_size_sample_small_batch_size_merge(self):
    num_seeds = 5000
    samples_per_seed = 30
    batch_size = 3
    res_len = 10
    margin_of_error = 0.05

    final_num_samples = num_seeds * res_len
    ideal_final_count = final_num_samples / samples_per_seed
    min_final_count = ideal_final_count * (1 - margin_of_error)
    max_final_count = ideal_final_count * (1 + margin_of_error)

    sample_histo = [0] * samples_per_seed
    for seed in range(num_seeds):
      sampler = rolling_stats.FixedSizeSample(
          max_size=res_len, _random_seed=seed
      )

      for batch in self._batch_data(range(samples_per_seed), batch_size):
        sampler.merge(
            rolling_stats.FixedSizeSample(
                max_size=res_len, _random_seed=seed
            ).add(batch)
        )

      for sample in sampler.result():
        sample_histo[int(sample)] += 1

    # Assert that all the reservoir samples were added to the histogram.
    self.assertEqual(sum(sample_histo), final_num_samples)

    # Assert that the histogram values are within the margin of error.
    self.assertTrue(all(bucket > min_final_count for bucket in sample_histo))
    self.assertTrue(all(bucket < max_final_count for bucket in sample_histo))


class RollingStateTest(parameterized.TestCase):

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
      if key == 'batch_score_fn':
        self.assertAlmostEqual(value, got[key])
      else:
        np.testing.assert_allclose(value, got[key])

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

  def test_mean_and_variance_invalid_batch_score_fn(self):
    batch = [1, 2, 3]
    state = rolling_stats.MeanAndVariance(batch_score_fn=lambda x: [0])
    with self.assertRaisesRegex(
        ValueError,
        'The `batch_score_fn` must return a series of the same length as the'
        ' `batch`.',
    ):
      state.add(batch)

  @parameterized.named_parameters([
      dict(
          testcase_name='merge_nonempty_states',
          is_self_empty=False,
          is_other_empty=False,
      ),
      dict(
          testcase_name='merge_nonempty_with_empty_state',
          is_self_empty=False,
          is_other_empty=True,
      ),
      dict(
          testcase_name='merge_empty_with_nonempty_state',
          is_self_empty=True,
          is_other_empty=False,
      ),
      dict(
          testcase_name='merge_empty_states',
          is_self_empty=True,
          is_other_empty=True,
      ),
      dict(
          testcase_name='merge_nonempty_multi_dim_states',
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
        np.append(self_batch, other_batch)
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
    agg_fn = rolling_stats.MeanAndVarianceAggFn()
    batches = np.arange(3)
    actual = agg_fn(batches)
    self.assertDataclassAlmostEqual(
        get_expected_mean_and_variance(batches), actual
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

    actual_result = rolling_stats.MinMaxAndCount(batch_score_fn=len).add(inputs)

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

    expected_result = r2_metric(
        sum_y_true=2.0, sum_y_pred=1.2, sum_neg_y_true=1.0, sum_neg_y_pred=0.8
    )

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

    actual_result = r2_metric().add(y_true, y_pred).result()

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


class RRegressionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='centered', center=True),
      dict(testcase_name='not_centered', center=False),
  )
  def test_r_regression_merge(self, center):
    x_1 = (1, 2, 3, 4)
    y_1 = (10, 9, 2.5, 6)

    x_2 = (5, 6, 7)
    y_2 = (4, 3, 2)

    state_1 = rolling_stats.RRegression(center=center).add(x_1, y_1)
    state_2 = rolling_stats.RRegression(center=center).add(x_2, y_2)
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

    actual_result = rolling_stats.RRegression(center=center).add(x, y).result()

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

    actual_result = (
        rolling_stats.SymmetricPredictionDifference().add(x, y).result()
    )

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
