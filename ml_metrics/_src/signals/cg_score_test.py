from absl.testing import absltest
from ml_metrics._src.signals import cg_score
import numpy as np


class CgScoreTest(absltest.TestCase):

  def test_validate_nonbinary_input(self):
    labels = ['A', 'B', 'C']
    with self.assertRaisesRegex(ValueError, 'only works for binary label'):
      cg_score.complexity_gap_score(labels, np.array([[1, 2], [2, 3], [5, 4]]))

  def test_group_data_by_label(self):
    input_embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    labels = np.array(['a', 'b', 'b'])
    result = cg_score._group_data_by_label(input_embeddings, labels)

    self.assertIn('a', result)
    self.assertSequenceAlmostEqual([0], result['a']['indices'])
    self.assertLen(result['a']['data'], 1)
    self.assertAlmostEqual(1, np.linalg.norm(result['a']['data']))

    self.assertIn('b', result)
    self.assertSequenceAlmostEqual([1, 2], result['b']['indices'])
    self.assertLen(result['b']['data'], 2)
    self.assertAlmostEqual(1, np.linalg.norm(result['b']['data'][0]))
    self.assertAlmostEqual(1, np.linalg.norm(result['b']['data'][1]))

  def test_get_other_label(self):
    example_dict = {
        'a': {
            'data': np.array([1, 2, 3]),
            'indices': [0, 1],
        },
        'b': {
            'data': np.array([4, 5, 6]),
            'indices': [0, 1],
        },
    }
    self.assertEqual(cg_score._get_other_label(example_dict, 'a'), 'b')
    self.assertEqual(cg_score._get_other_label(example_dict, 'b'), 'a')

  def test_balance_dataset(self):
    data_a = np.array([[1, 2, 3]])
    data_b = np.array(np.array([[4, 5, 6], [7, 8, 9]]))
    self.assertLen(cg_score._balance_dataset(data_a, data_b, 1), 2)
    self.assertLen(cg_score._balance_dataset(data_a, data_b, 2), 3)
    self.assertLen(cg_score._balance_dataset(data_b, data_a, 1), 3)

  def test_calculate_complexity_gap_score_result_sanity(self):
    embeddings = np.array([[1, 2], [2, 3], [5, 4], [5, 4]])
    labels = np.array([1, 1, 1, 0])
    cg_scores = cg_score.complexity_gap_score(
        labels, embeddings, num_repetitions=1
    )
    self.assertLen(cg_scores, 4)
    self.assertLess(cg_scores[0], cg_scores[3])

  def test_calculate_complexity_gap_score_simple(self):
    embeddings = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    labels = np.array([1, 1, 0, 0])
    cg_scores = cg_score.complexity_gap_score(
        labels, embeddings, num_repetitions=1
    )
    self.assertSequenceAlmostEqual(cg_scores, [0, 0, 0, 0])

  def test_num_repetitions(self):
    embeddings = np.array([[1, 2], [1, 2], [5, 4], [5, 4]])
    labels = np.array([1, 1, 0, 0])
    cg_scores_one = cg_score.complexity_gap_score(
        labels, embeddings, num_repetitions=1
    )
    cg_scores_three = cg_score.complexity_gap_score(
        labels, embeddings, num_repetitions=3
    )
    self.assertLen(cg_scores_one, 4)
    self.assertLen(cg_scores_three, 4)
    np.testing.assert_allclose(cg_scores_one, cg_scores_three)


if __name__ == '__main__':
  absltest.main()
