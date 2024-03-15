"""Courier server tests."""

import pickle
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import runner_utils
from absl.testing import absltest


class CustomPickler:

  def dumps(self, x):
    return pickle.dumps(x)

  def loads(self, x):
    return pickle.loads(x)


class _SumAggFn:
  """Mock CombineFn for test."""

  def create_state(self):
    return 0

  def update_state(self, state, x):
    return state + sum(x)

  def merge_states(self, states):
    return sum(states)

  def get_result(self, state):
    return state


class RunnerUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    custom_pickler = CustomPickler()
    runner_utils.picklers.register(custom_pickler)
    self.pickler = runner_utils.picklers.default

  def test_maybe_make(self):
    self.assertEqual(3, runner_utils.maybe_make(lazy_fns.trace(len)([1, 2, 3])))
    pickled = self.pickler.dumps(lazy_fns.trace(len)([1, 2, 3]))
    self.assertEqual(3, runner_utils.maybe_make(pickled))

  def test_pickler_register_assertion(self):
    with self.assertRaises(TypeError):
      runner_utils.picklers.register(len)


if __name__ == '__main__':
  absltest.main()
