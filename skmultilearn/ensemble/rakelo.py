from .rakeld import RakelD
import copy
import numpy as np
import random
import six
from scipy import sparse

class RakelO(RakelD):
    """

    Overlapping RAndom k-labELsets multi-label classifier.

    """

    def __init__(self, classifier = None, models = None, labelset_size = None, require_dense = None):
        super(RakelO, self).__init__(classifier = classifier, require_dense = require_dense)
        self.model_count = int(models)
        self.labelset_size = labelset_size

    def generate_partition(self, X, y):
        """Internal method for sampling k-labELsets"""
        label_sets = []
        self.label_count = y.shape[1]
        free_labels = six.moves.range(self.label_count)
        
        while len(label_sets) < self.model_count:
            label_set = random.sample(free_labels, self.labelset_size)
            if label_set not in label_sets:
                label_sets.append(label_set)

        self.partition = label_sets
        assert len(self.partition) == self.model_count

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        predictions = [
            self.ensure_input_format(self.ensure_input_format(c.predict(X)), sparse_format = 'csc', enforce_sparse = True)
            for c in self.classifiers
        ]

        votes = sparse.csc_matrix((predictions[0].shape[0], self.label_count), dtype='i8')
        for model in six.moves.range(self.model_count):
            for label in six.moves.range(len(self.partition[model])):
                votes[:, self.partition[model][label]] = votes[:, self.partition[model][label]]  + predictions[model][:, label]

        voters = map(float, votes.sum(axis = 0).tolist()[0])

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = votes[row, column] / voters[column]

        return votes
