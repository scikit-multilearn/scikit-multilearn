from builtins import range
from ..base import MLClassifierBase
from ..utils import get_matrix_in_format

from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sparse


class MLkNN(MLClassifierBase):
    """kNN classification method adapted for multi-label classification

    Parameters
    ----------

    k : integer
        number of neighbours of each input instance to take into account

    s: boolean
            the smoothing parameter

    ignore_first_neighbours : integer
            ability to ignore first N neighbours, useful for comparing with other classification software, if you
            don't know what it does, the default is safe, see https://github.com/scikit-multilearn/scikit-multilearn/issues/22

    """
    BRIEFNAME = "MLkNN"

    def __init__(self, k=10, s=1.0, ignore_first_neighbours=0):
        super(MLkNN, self).__init__()
        self.k = k  # Number of neighbours
        self.s = s  # Smooth parameter
        self.ignore_first_neighbours = ignore_first_neighbours
        self.copyable_attrs = ['k', 's', 'ignore_first_neighbours']

    def compute_prior(self, y):
        prior_prob_true = np.array((self.s + y.sum(axis=0)) / (self.s * 2 + self.num_instances))[0]
        prior_prob_false = 1 - prior_prob_true

        return prior_prob_true, prior_prob_false

    def compute_cond(self, X, y):
        self.knn = NearestNeighbors(self.k).fit(X)
        c = sparse.lil_matrix((self.num_labels, self.k + 1), dtype='i8')
        cn = sparse.lil_matrix((self.num_labels, self.k + 1), dtype='i8')

        label_info = get_matrix_in_format(y, 'dok')

        neighbors = [a[self.ignore_first_neighbours:] for a in
                     self.knn.kneighbors(X, self.k + self.ignore_first_neighbours, return_distance=False)]

        for instance in range(self.num_instances):
            deltas = label_info[neighbors[instance], :].sum(axis=0)
            for label in range(self.num_labels):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix((self.num_labels, self.k + 1), dtype='float')
        cond_prob_false = sparse.lil_matrix((self.num_labels, self.k + 1), dtype='float')
        for label in range(self.num_labels):
            for neighbor in range(self.k + 1):
                cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                self.s * (self.k + 1) + c_sum[label, 0])
                cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                self.s * (self.k + 1) + cn_sum[label, 0])
        return cond_prob_true, cond_prob_false

    def fit(self, X, y):
        self.train_labels = get_matrix_in_format(y, 'lil')
        self.num_instances = self.train_labels.shape[0]
        self.num_labels = self.train_labels.shape[1]
        # Computing the prior probabilities
        self.prior_prob_true, self.prior_prob_false = self.compute_prior(self.train_labels)
        # Computing the posterior probabilities
        self.cond_prob_true, self.cond_prob_false = self.compute_cond(X, self.train_labels)
        return self

    def predict(self, X):
        result = sparse.lil_matrix((X.shape[0], self.num_labels), dtype='i8')
        neighbors = [a[self.ignore_first_neighbours:] for a in
                     self.knn.kneighbors(X, self.k + self.ignore_first_neighbours, return_distance=False)]
        for instance in range(X.shape[0]):
            deltas = self.train_labels[neighbors[instance],].sum(axis=0)

            for label in range(self.num_labels):
                p_true = self.prior_prob_true[label] * self.cond_prob_true[label, deltas[0, label]]
                p_false = self.prior_prob_false[label] * self.cond_prob_false[label, deltas[0, label]]
                result[instance, label] = int(p_true >= p_false)
        return result

    def predict_proba(self, X):
        result = sparse.lil_matrix((X.shape[0], self.num_labels), dtype='float')
        neighbors = [a[self.ignore_first_neighbours:] for a in
                     self.knn.kneighbors(X, self.k + self.ignore_first_neighbours, return_distance=False)]
        for instance in range(X.shape[0]):
            deltas = self.train_labels[neighbors[instance],].sum(axis=0)

            for label in range(self.num_labels):
                p_true = self.prior_prob_true[label] * self.cond_prob_true[label, deltas[0, label]]
                p_false = self.prior_prob_false[label] * self.cond_prob_false[label, deltas[0, label]]
                result[instance, label] = p_true
        return result

