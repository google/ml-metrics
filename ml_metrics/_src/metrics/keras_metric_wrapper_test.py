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
"""Tests for Classification evaluation metrics API."""

from absl.testing import absltest
from absl.testing import parameterized
from ml_metrics._src.metrics import keras_metric_wrapper
import numpy as np


class KerasMetricWrapperTest(parameterized.TestCase):

  def test_roc_auc(self):
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0.5, 0.3, 0.9]
    np.testing.assert_allclose(
        (3 / 4,),
        keras_metric_wrapper.roc_auc(y_true, y_pred),
    )

if __name__ == "__main__":
  absltest.main()
