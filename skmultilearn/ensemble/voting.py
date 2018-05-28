from builtins import zip
from builtins import map
from builtins import range
from .partition import LabelSpacePartitioningClassifier
import copy
import numpy as np
import random
from scipy import sparse


class MajorityVotingClassifier(LabelSpacePartitioningClassifier):
    """Overlapping RAndom k-labELsets multi-label classifier"""

    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        super(MajorityVotingClassifier, self).__init__(
            classifier=classifier, clusterer=clusterer, require_dense=require_dense)

    def predict(self, X):
        """Predict probabilities of label assignments for X

        Internally this method uses a sparse CSC representation for X
        (:class:`scipy.sparse.csr_matrix`).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of float
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """
        predictions = [
            self.ensure_input_format(self.ensure_input_format(
                c.predict(X)), sparse_format='csc', enforce_sparse=True)
            for c in self.classifiers
        ]

        voters = np.zeros(self.label_count, dtype='int')
        votes = sparse.csc_matrix(
            (predictions[0].shape[0], self.label_count), dtype='int')
        for model in range(self.model_count):
            for label in range(len(self.partition[model])):
                votes[:, self.partition[model][label]] = votes[
                    :, self.partition[model][label]] + predictions[model][:, label]
                voters[self.partition[model][label]]+=1

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = np.round(
                votes[row, column] / float(voters[column]))

        return self.ensure_output_format(votes, enforce_sparse=False)
