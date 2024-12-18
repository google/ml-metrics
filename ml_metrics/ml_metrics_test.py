"""Tests for ml_metrics import."""

from absl.testing import absltest


class MlMetricsTest(absltest.TestCase):

  def test_import_ml_metrics(self):
    try:
      # Only test the importing of the module succeeds.
      from ml_metrics import ml_metrics  # pylint: disable=g-import-not-at-top
    except Exception:  # pylint: disable=broad-exception-caught
      self.fail("Failed to import ml_metrics. Fix the dependency.")
    self.assertIsNotNone(ml_metrics)


if __name__ == "__main__":
  absltest.main()
