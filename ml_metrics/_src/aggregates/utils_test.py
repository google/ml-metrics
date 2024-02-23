"""Unit tests for utils.py."""

import collections.abc
from ml_metrics._src.aggregates import utils
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('both_zero', 0, 0, 0.0),
      ('zero_num', 0, 10, 0.0),
      ('zero_denom', 10, 0, 0.0),
      ('float_num', 10.5, 3, 3.5),
      ('float_denom', 14, 3.5, 4.0),
      ('array_num', [2, 4, 6, 8], 2, [1.0, 2.0, 3.0, 4.0]),
      ('array_num_denom', [2, 4, 6, 8], [4, 8, 12, 16], [0.5, 0.5, 0.5, 0.5]),
  )
  def test_safe_divide(self, a, b, expected_result):
    result = utils.safe_divide(a, b)
    if isinstance(result, (collections.abc.Sequence, np.ndarray)):
      self.assertSequenceAlmostEqual(result, expected_result)
    else:
      self.assertAlmostEqual(result, expected_result)

  def test_pos_sqrt(self):
    self.assertEqual(utils.pos_sqrt(4), 2.0)
    with self.assertRaisesRegex(
        ValueError, 'Attempt to take sqrt of negative value: -1'
    ):
      utils.pos_sqrt(-1)


if __name__ == '__main__':
  absltest.main()
