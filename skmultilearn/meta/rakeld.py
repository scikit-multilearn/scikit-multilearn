from ..base import MLClassifierBase
import copy
import random
import numpy as np
from scipy import sparse

class RakelD(MLClassifierBase):
    """Distinct RAndom k-labELsets multi-label classifier."""

    def __init__(self, classifier = None, labelset_size = None):
        super(RakelD, self).__init__(classifier)
        self.labelset_size = labelset_size

    def sample_models(self, label_count = None):
        if label_count is not None:
            self.label_count = label_count

        """Internal method for sampling k-labELsets"""
        label_sets = []
        free_labels = xrange(self.label_count)
        self.model_count = int(np.ceil(self.label_count/self.labelset_size))

        while len(label_sets) <= self.model_count:
            if len(free_labels) == 0:
                break
            if len(free_labels) < self.labelset_size:
                label_sets.append(free_labels)
                continue

            label_set = random.sample(free_labels, self.labelset_size)
            free_labels = list(set(free_labels).difference(set(label_set)))
            label_sets.append(label_set)

        self.label_sets = label_sets

    def fit_only(self, X, y):
        """Fit classifier according to X,y, without resampling models."""
        self.classifiers = []
        for i in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y,self.label_sets[i])
            classifier.fit(X,y_subset)
            self.classifiers.append(classifier)

        return self

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.sample_models(y.shape[1])
        return self.fit_only(X, y)

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        input_rows = X.shape[0]
        predictions = [self.classifiers[i].predict(X) for i in xrange(self.model_count)]
        result = sparse.lil_matrix((input_rows, self.label_count), dtype=int)

        for model in xrange(self.model_count):
            predictions = self.classifiers[model].predict(X).nonzero()
            for row, column in zip(predictions[0], predictions[1]):
                result[row, self.label_sets[model][column]] = 1

        return result
