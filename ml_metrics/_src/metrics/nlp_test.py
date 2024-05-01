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
"""Tests for nlp."""

from ml_metrics._src.metrics import nlp
from absl.testing import absltest


class NlpTest(absltest.TestCase):

  def test_avg_char_count(self):
    texts = ['abc', 'd e!', None]
    result = nlp.avg_char_count(texts)
    self.assertAlmostEqual(float(5/2), result)

  def test_avg_char_count_empty(self):
    result = nlp.avg_char_count([])
    self.assertAlmostEqual(float(0), result)

if __name__ == '__main__':
  absltest.main()
