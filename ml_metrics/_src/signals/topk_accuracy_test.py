"""Unit tests for topk_accuracy.py."""

from ml_metrics._src.signals import topk_accuracy
from absl.testing import absltest
from absl.testing import parameterized


class TopkAccuracyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="top_k_accuracy_label1_k1",
          y_pred=[0.2, 0.7, 0.1],
          label=1,
          k=1,
          expected=1.0,
      ),
      dict(
          testcase_name="top_k_accuracy_label0_k1",
          y_pred=[0.2, 0.7, 0.1],
          label=0,
          k=1,
          expected=0.0,
      ),
      dict(
          testcase_name="top_k_accuracy_label0_k2",
          y_pred=[0.2, 0.7, 0.1],
          label=0,
          k=2,
          expected=1.0,
      ),
  )
  def test_topk_accuracy(self, y_pred, label, k, expected):
    self.assertEqual(topk_accuracy.topk_accurate(y_pred, label, k), expected)


if __name__ == "__main__":
  absltest.main()
