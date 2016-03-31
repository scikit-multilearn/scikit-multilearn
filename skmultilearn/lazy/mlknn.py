from ..base import MLClassifierBase
from sklearn.neighbors import NearestNeighbors
import numpy as np
import six

class KNearestNeighbours(MLClassifierBase):
    """k Nearest Neighbours multi-label classifier."""
    BRIEFNAME = "MLkNN"

    def __init__(self, k = 10, s = 1.0):
        super(KNearestNeighbours, self).__init__(None)
        self.k = k # Number of neighbours
        self.s = s # Smooth parameter

    def compute_prior(self, y):
        prior_prob_true = []
        prior_prob_false = []
        for label in six.moves.range(self.num_labels):
            prior_prob_true.append(float(self.s + sum(instance[label] == 1 for instance in y)) / (self.s * 2 + self.num_instances))
            prior_prob_false.append(1 - prior_prob_true[-1])
        return prior_prob_true, prior_prob_false

    def compute_cond(self, X, y):
        self.knn = NearestNeighbors(self.k).fit(X)
        c = [[0] * (self.k + 1) for label in six.moves.range(self.num_labels)]
        cn = [[0] * (self.k + 1) for label in six.moves.range(self.num_labels)]
        for instance in six.moves.range(self.num_instances):
            neighbors = self.knn.kneighbors(X[instance], self.k, return_distance=False)
            for label in six.moves.range(self.num_labels):
                delta = sum(y[neighbor][label] for neighbor in neighbors[0])
                (c if y[instance][label] == 1 else cn)[label][delta] += 1

        cond_prob_true = [[0] * (self.k + 1) for label in six.moves.range(self.num_labels)]
        cond_prob_false = [[0] * (self.k + 1) for label in six.moves.range(self.num_labels)]
        for label in six.moves.range(self.num_labels):
            for neighbor in six.moves.range(self.k + 1):
                cond_prob_true[label][neighbor] = (self.s + c[label][neighbor]) / (self.s * (self.k + 1) + sum(c[label]))
                cond_prob_false[label][neighbor] = (self.s + cn[label][neighbor]) / (self.s * (self.k + 1) + sum(cn[label]))
        return cond_prob_true, cond_prob_false

    def fit(self, X, y):
        self.predictions = y;
        self.num_instances = len(y)
        self.num_labels = len(y[0])
        # Computing the prior probabilities
        self.prior_prob_true, self.prior_prob_false = self.compute_prior(y)
        # Computing the posterior probabilities
        self.cond_prob_true, self.cond_prob_false = self.compute_cond(X, y)
        return self

    def predict(self, X):
        result = np.zeros((len(X), self.num_labels), dtype='i8')
        for instance in six.moves.range(len(X)):
            neighbors = self.knn.kneighbors(X[instance], self.k, return_distance=False)
            for label in six.moves.range(self.num_labels):
                delta = sum(self.predictions[neighbor][label] for neighbor in neighbors[0])
                p_true = self.prior_prob_true[label] * self.cond_prob_true[label][delta]
                p_false = self.prior_prob_false[label] * self.cond_prob_false[label][delta]
                prediction = (p_true >= p_false)
                result[instance][label] = int(prediction)
        return result
