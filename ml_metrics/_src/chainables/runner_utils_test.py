"""Courier server tests."""

import pickle
from ml_metrics._src.chainables import lazy_fns
from ml_metrics._src.chainables import runner_utils
from ml_metrics._src.chainables import transform
from ml_metrics._src.chainables import tree
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

  def test_run_update_state(self):
    t = transform.AggregateTransform.new().add_aggregate(
        output_keys='result',
        fn=lazy_fns.trace(_SumAggFn)(),
        input_keys=tree.Key.SELF,
    )
    inputs = [[0, 1, 2], [1, 2, 3]]

    v = runner_utils.run_update_state(t, inputs)
    self.assertEqual({t.fns[0]: 9}, self.pickler.loads(v))

  def test_run_update_state_with_num_steps(self):
    t = transform.AggregateTransform.new().add_aggregate(
        output_keys='result',
        fn=lazy_fns.trace(_SumAggFn)(),
        input_keys=tree.Key.SELF,
    )
    inputs = [[0, 1, 2], [1, 2, 3], ['wrong', 'input']]
    num_steps, states, busy = [-1], [None], [False]

    def record_step(i):
      num_steps[0] = i

    def record_state(i, state):
      states[0] = i, state

    def record_busy(i):
      busy[0] = i

    with self.assertRaises(ValueError):
      runner_utils.run_update_state(
          self.pickler.dumps(t),
          self.pickler.dumps(inputs),
          num_steps=3,
          record_num_steps=record_step,
          record_state=record_state,
          record_busy=record_busy,
      )
    self.assertEqual(2, num_steps[0])
    self.assertEqual((1, {t.fns[0]: 9}), states[0])
    self.assertEqual(True, busy[0])

  def test_run_merge_states(self):
    t = transform.AggregateTransform.new().add_aggregate(
        output_keys='result',
        fn=lazy_fns.trace(_SumAggFn)(),
        input_keys=tree.Key.SELF,
    )
    inputs = [[0, 1, 2], [1, 2, 3]]
    t1 = t.make()
    t2 = t.make()
    state1 = t1.update_state(t1.create_state(), inputs[0])
    state2 = t2.update_state(t2.create_state(), inputs[1])
    v = runner_utils.run_merge_states(t, [state1, state2])
    self.assertEqual({t.fns[0]: 9}, self.pickler.loads(v))

  def test_run_get_result(self):
    t = transform.AggregateTransform.new().add_aggregate(
        output_keys='result',
        fn=lazy_fns.trace(_SumAggFn)(),
        input_keys=tree.Key.SELF,
    )
    inputs = [0, 1, 2]
    state = t.make().update_state(t.make().create_state(), inputs)
    v = runner_utils.run_get_result(t, state)
    self.assertEqual({'result': 3}, self.pickler.loads(v))

  def test_pickler_register_assertion(self):
    with self.assertRaises(TypeError):
      runner_utils.picklers.register(len)


if __name__ == '__main__':
  absltest.main()
