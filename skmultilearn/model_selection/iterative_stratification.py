from sklearn.model_selection._split import _BaseKFold
import numpy as np
import scipy.sparse as sp
import itertools


class IterativeStratification(_BaseKFold):

    def __init__(self, n_splits=3, order=1, random_state=None):
        self.order = order
        super(
            IterativeStratification,
            self).__init__(n_splits,
                           shuffle=False,
                           random_state=random_state)

    def init_values(self, X):
        self.n_samples = X.shape[0]
        self.n_labels = X.shape[1]
        self.percentage_per_fold = [1 / float(self.n_splits)
                                    for i in range(self.n_splits)]
        self.desired_samples_per_fold = np.array([self.percentage_per_fold[i] * self.n_samples
                                                  for i in range(self.n_splits)])
        self.rows = sp.lil_matrix(X).rows
        self.rows_used = {i: False for i in range(self.n_samples)}

        self.init_row_and_label_data(X)

    def init_row_and_label_data(self, X):
        self.all_combinations = []
        self.per_row_combinations = [[] for i in range(self.n_samples)]
        self.samples_with_combination = {}
        for i, x in enumerate(self.rows):
            for y in itertools.combinations_with_replacement(x, self.order):
                if y not in self.samples_with_combination:
                    self.samples_with_combination[y] = []

                self.samples_with_combination[y].append(i)

                self.all_combinations.append(y)
                self.per_row_combinations[i].append(y)

        self.all_combinations = [list(x) for x in set(self.all_combinations)]

        self.desired_samples_per_label_per_fold = {
            i:
               [len(self.samples_with_combination[i]) * self.percentage_per_fold[j]
                for j in range(self.n_splits)]
                for i in self.samples_with_combination
        }

    def label_tie_break(self, M):
        if len(M) == 1:
            return M[0]
        else:
            max_val = max(self.desired_samples_per_fold[M])
            M_prim = np.where(
                np.array(self.desired_samples_per_fold) == max_val)[0]
            M_prim = np.array([x for x in M_prim if x in M])
            return np.random.choice(M_prim, 1)[0]

    def get_most_desired_combination(self):
        currently_chosen = None
        currently_best_score = None
        for combination, evidence in self.samples_with_combination.items():
            current_score = (len(set(combination)), len(evidence))
            if current_score[1] == 0:
                continue
            if currently_chosen is None:
                currently_chosen = combination
                currently_best_score = current_score
                continue

            if currently_best_score[1] > current_score[1] and currently_best_score[0] < current_score[0]:
                print(currently_best_score, current_score)
                currently_chosen = combination
                currently_best_score = current_score

        if current_score is not None:
            return currently_chosen

        return None

    def distribute_positive_evidence(self):
        l = self.get_most_desired_combination()
        while l is not None:
            print(l)
            while len(self.samples_with_combination[l]) > 0:
                row = self.samples_with_combination[l].pop()
                print(len(self.samples_with_combination[l]), l, row)
                if self.rows_used[row]:
                    continue

                max_val = max(self.desired_samples_per_label_per_fold[l])
                M = np.where(
                    np.array(self.desired_samples_per_label_per_fold[l]) == max_val)[0]
                m = self.label_tie_break(M)
                self.folds[m].append(row)
                self.rows_used[row] = True
                for i in self.per_row_combinations[row]:
                    if row in self.samples_with_combination[i]:
                        self.samples_with_combination[i].remove(row)
                    self.desired_samples_per_label_per_fold[i][m] -= 1
                self.desired_samples_per_fold[m] -= 1

            l = self.get_most_desired_combination()

    def distribute_negative_evidence(self):
        self.available_samples = [
            i for i, v in self.rows_used.items() if not v]
        self.samples_left = len(self.available_samples)

        while self.samples_left > 0:
            row = self.available_samples.pop()
            self.rows_used[row] = True
            self.samples_left -= 1
            fold_selected = np.random.choice(
                np.where(self.desired_samples_per_fold > 0)[0],
                1)[0]
            self.desired_samples_per_fold[fold_selected] -= 1
            self.folds[fold_selected].append(row)

    def _iter_test_indices(self, X, y=None, groups=None):
        self.init_values(X)
        if self.shuffle:
            check_random_state(self.random_state)

        self.folds = [[] for i in range(self.n_splits)]
        self.distribute_positive_evidence()
        self.distribute_negative_evidence()

        for fold in self.folds:
            yield fold
