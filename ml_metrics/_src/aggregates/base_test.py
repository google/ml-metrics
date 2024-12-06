# Copyright 2023 Google LLC
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
"""Tests for core lib."""

from absl.testing import absltest
from ml_metrics._src.aggregates import base
from ml_metrics._src.aggregates import test_utils
from ml_metrics._src.chainables import lazy_fns


class AggregatesTest(absltest.TestCase):

  def test_callable_combinefn_in_process(self):
    sum_fn = base.UserAggregateFn(test_utils._SumAggFn())
    self.assertEqual(sum_fn(list(range(4))), 6)

  def test_mergeable_aggregate_fn_in_process(self):
    sum_fn = base.as_agg_fn(test_utils._SumMetric)
    self.assertEqual(6, sum_fn([1, 2, 3]))

  def test_mergeable_aggregate_fn_from_resolvable(self):
    makeable_deferred_sum = lazy_fns.trace(test_utils._SumMetric)()
    sum_fn = base.MergeableMetricAggFn(makeable_deferred_sum)
    self.assertEqual(6, sum_fn([1, 2, 3]))

  def test_mergeable_aggregate_fn_unsupported_type(self):
    with self.assertRaisesRegex(TypeError, 'must be an instance of.+ got'):
      # disable pytype check for the runtime error to surface.
      _ = base.MergeableMetricAggFn(test_utils._SumMetric())  # pytype: disable=wrong-arg-types

  def test_metric_callable(self):
    class Sum(test_utils._SumMetric, base.CallableMetric):
      pass

    sum_fn = Sum()
    self.assertEqual(6, sum_fn([1, 2, 3]))


if __name__ == '__main__':
  absltest.main()
