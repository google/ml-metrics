# Copyright 2025 Google LLC
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
from ml_metrics._src.utils import test_utils
from absl.testing import absltest


class TestUtilsTest(absltest.TestCase):

  def test_inf_range(self):
    with self.assertRaises(ValueError):
      _ = list(test_utils.inf_range(10))

  def test_nolen_iter(self):
    with self.assertRaises(ValueError):
      if test_utils.NoLenIter(range(10)):
        pass


if __name__ == "__main__":
  absltest.main()
