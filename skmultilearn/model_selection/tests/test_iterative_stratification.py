import unittest
import numpy as np
from skmultilearn.model_selection import IterativeStratification


class IterativeStratificationTest(unittest.TestCase):
    def test_if_variables_are_initialized_correctly(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])

        (
            rows,
            rows_used,
            all_combinations,
            per_row_combinations,
            samples_with_combination,
            folds,
        ) = stratifier._prepare_stratification(y)

        self.assertEqual(stratifier.n_samples, 4)
        self.assertEqual(stratifier.n_labels, 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(rows_used), 4)
        self.assertEqual(len(stratifier.percentage_per_fold), 2)
        self.assertEqual(len(stratifier.desired_samples_per_fold), 2)
        self.assertEqual(len(folds), 2)
        self.assertTrue(not any(rows_used.values()))
        self.assertFalse(any(rows_used.values()))
        self.assertEqual(stratifier.order, 1)

        for d in stratifier.percentage_per_fold:
            self.assertEqual(d, 1 / 2.0)

        for d in stratifier.desired_samples_per_fold:
            self.assertEqual(d, y.shape[0] / 2.0)

        self.assertEqual(len(all_combinations), 2)
        self.assertEqual(len(per_row_combinations[0]), 0)
        self.assertEqual(len(per_row_combinations[1]), 1)
        self.assertEqual(len(per_row_combinations[2]), 1)
        self.assertEqual(len(per_row_combinations[3]), 2)

        self.assertEqual(len(samples_with_combination), 2)
        self.assertEqual(len(stratifier.desired_samples_per_combination_per_fold), 2)
        for combination, samples in samples_with_combination.items():
            self.assertEqual(len(set(combination)), 1)
            self.assertEqual(len(samples), 2)

        for (
            combination,
            desirability,
        ) in stratifier.desired_samples_per_combination_per_fold.items():
            self.assertEqual(len(set(combination)), 1)
            self.assertEqual(len(desirability), 2)
            for desire in desirability:
                self.assertEqual(desire, 1.0)

    def test_if_positive_evidence_does_not_include_negative_evidence(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])

        (
            rows,
            rows_used,
            all_combinations,
            per_row_combinations,
            samples_with_combination,
            folds,
        ) = stratifier._prepare_stratification(y)

        stratifier._distribute_positive_evidence(
            rows_used, folds, samples_with_combination, per_row_combinations
        )
        self.assertFalse(rows_used[0])
        self.assertTrue(rows_used[1])
        self.assertTrue(rows_used[2])
        self.assertTrue(rows_used[3])

        for (
            combination,
            samples,
        ) in stratifier.desired_samples_per_combination_per_fold.items():
            for desire in samples:
                self.assertEqual(desire, 0)

    def test_if_negative_evidence_is_distributed(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])

        (
            rows,
            rows_used,
            all_combinations,
            per_row_combinations,
            samples_with_combination,
            folds,
        ) = stratifier._prepare_stratification(y)

        stratifier._distribute_positive_evidence(
            rows_used, folds, samples_with_combination, per_row_combinations
        )
        self.assertFalse(rows_used[0])

        stratifier._distribute_negative_evidence(rows_used, folds)
        self.assertTrue(rows_used[0])

    def test_if_stratification_works(self):
        stratifier = IterativeStratification(n_splits=2, order=1)
        X = np.matrix([[0], [1], [2], [3]])
        y = np.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.assertEqual(len(list(stratifier.split(X, y))), 2)


if __name__ == "__main__":
    unittest.main()
