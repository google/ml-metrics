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


class CoreTest(absltest.TestCase):

  def test_callable_combinefn_in_process(self):
    sum_fn = base.UserAggregateFn(test_utils._SumAggFn())
    self.assertEqual(sum_fn(list(range(4))), 6)


if __name__ == "__main__":
  absltest.main()
