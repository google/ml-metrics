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
"""Tests for Cross Entropy Metrics."""

from ml_metrics._src.scores import cross_entropy_metrics
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses

from absl.testing import absltest
from absl.testing import parameterized


_PLACES = 6  # Used to tune tests accuracy.


class CrossEntropyMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='default', from_logits=False, label_smoothing=0),
      dict(testcase_name='from_logits', from_logits=True, label_smoothing=0),
      dict(
          testcase_name='label_smoothing',
          from_logits=False,
          label_smoothing=0.5,
      ),
      dict(
          testcase_name='from_logits_and_label_smoothing',
          from_logits=True,
          label_smoothing=0.5,
      ),
  )
  def test_binary_cross_entropy(self, from_logits, label_smoothing):
    y_true = np.array((0, 1, 0, 1, 0, 1, 0, 1))
    y_pred = np.array((0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9))

    self.assertAlmostEqual(
        cross_entropy_metrics.binary_cross_entropy(
            y_true=y_true,
            y_pred=y_pred,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        ),
        tf.get_static_value(
            losses.BinaryCrossentropy(
                from_logits=from_logits, label_smoothing=label_smoothing
            )(y_true=y_true, y_pred=y_pred)
        ),
        places=_PLACES,
    )

  @parameterized.named_parameters(
      dict(testcase_name='default', from_logits=False, label_smoothing=0),
      dict(testcase_name='from_logits', from_logits=True, label_smoothing=0),
      dict(
          testcase_name='label_smoothing',
          from_logits=False,
          label_smoothing=0.5,
      ),
      dict(
          testcase_name='from_logits_and_label_smoothing',
          from_logits=True,
          label_smoothing=0.5,
      ),
  )
  def test_categorical_cross_entropy(self, from_logits, label_smoothing):
    y_true = np.array((0, 1, 0, 1, 0, 1, 0, 1))
    y_pred = np.array((0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9))

    self.assertAlmostEqual(
        cross_entropy_metrics.categorical_cross_entropy(
            y_true=y_true,
            y_pred=y_pred,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        ),
        tf.get_static_value(
            losses.CategoricalCrossentropy(
                from_logits=from_logits, label_smoothing=label_smoothing
            )(y_true=y_true, y_pred=y_pred)
        ),
        places=_PLACES,
    )


if __name__ == '__main__':
  absltest.main()
