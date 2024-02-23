"""Tests for ML metrics API function utils."""

from ml_metrics._src.aggregates import types
from ml_metrics._src.metrics import utils

from absl.testing import absltest


InputType = types.InputType
AverageType = types.AverageType


class UtilsTest(absltest.TestCase):

  def test_validate_inputs(self):
    with self.assertRaisesRegex(
        ValueError,
        r"Pos label 1 not in labels. Please set a valid pos_label from: \['n',"
        r" 'y'\]",
    ):
      _ = utils.verify_input(
          y_true=["y", "n", "n"],
          y_pred=["y", "n", "y"],
          average=AverageType.BINARY,
          input_type=InputType.BINARY,
          pos_label=1,
          vocab=None,
      )


if __name__ == "__main__":
  absltest.main()
