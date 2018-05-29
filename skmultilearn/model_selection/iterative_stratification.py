"""Iteratively stratify a multi-label data set

    The classifier follows methods outlined in Sechidis11 and Szymanski17 papers related to stratyfing
    multi-label data. 

    In general what we expect from a given stratification output is that a strata, or a fold, is close to a given, demanded size,
    usually equal to 1/k in k-fold approach, or a x% train to test set division in 2-fold splits. 

    The idea behind this stratification method is to assign label combinations to folds based on how much a given combination is desired
    by a given fold, as more and more assignments are made, some folds are filled and positive evidence is directed into other folds,
    in the end negative evidence is distributed based on a folds desirability of size.

    You can also watch a `video presentation <http://videolectures.net/ecmlpkdd2011_tsoumakas_stratification/?q=stratification%20multi%20label>`_ by G. Tsoumakas which explains the algorithm. In 2017 Szymanski & Kajdanowicz extended the algorithm
    to handle high-order relationships in the data set, if order = 1, the algorithm falls back to the original Sechidis11 setting. 

    If order is larger than 1 this class constructs a list of label combinations with replacement, i.e. allowing combinations of lower
    order to be take into account. For example for combinations of order 2, the stratifier will consider both
    label pairs (1, 2) and single labels denoted as (1,1) in the algorithm. In higher order cases the 
    when two combinations of different size have similar desirablity: the larger, i.e. more specific combination
    is taken into consideration first, thus if a label pair (1,2) and label 1 represented as (1,1) are of similar
    desirability, evidence for (1,2) will be assigned to folds first.


    If you use this method to stratify data please cite both:
    Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data. Machine Learning and Knowledge Discovery in Databases, 145-158.
    http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

    Piotr Szymański, Tomasz Kajdanowicz ; Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017.
    http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html

    Bibtex:

    .. code-block:: bibtex

        @article{sechidis2011stratification,
          title={On the stratification of multi-label data},
          author={Sechidis, Konstantinos and Tsoumakas, Grigorios and Vlahavas, Ioannis},
          journal={Machine Learning and Knowledge Discovery in Databases},
          pages={145--158},
          year={2011},
          publisher={Springer}
        }

        @InProceedings{pmlr-v74-szymański17a,
          title =    {A Network Perspective on Stratification of Multi-Label Data},
          author =   {Piotr Szymański and Tomasz Kajdanowicz},
          booktitle =    {Proceedings of the First International Workshop on Learning with Imbalanced Domains: Theory and Applications},
          pages =    {22--35},
          year =     {2017},
          editor =   {Luís Torgo and Bartosz Krawczyk and Paula Branco and Nuno Moniz},
          volume =   {74},
          series =   {Proceedings of Machine Learning Research},
          address =      {ECML-PKDD, Skopje, Macedonia},
          publisher =    {PMLR},
        }
"""



from sklearn.model_selection._split import _BaseKFold
import numpy as np
import scipy.sparse as sp
import itertools


class IterativeStratification(_BaseKFold):
    """Iteratively stratify a multi-label data set

    Construct an interative stratifier that conducts:

    :param n_splits: with number of splits
    :type n_splits: int

    :param order: taking into account order-th order of relationship
    :type order: int

    :param random_state: and the random state seed (optional)
    :type random_state: int

    """
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
        
        self.folds = [[] for i in range(self.n_splits)]
        
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
        if self.random_state:
            check_random_state(self.random_state)

        self.folds = [[] for i in range(self.n_splits)]
        self.distribute_positive_evidence()
        self.distribute_negative_evidence()

        for fold in self.folds:
            yield fold
