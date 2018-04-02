from builtins import zip
from builtins import map
from builtins import range
from .rakeld import RakelD
import copy
import numpy as np
import random
from scipy import sparse


class RakelO(RakelD):
    """Overlapping RAndom k-labELsets multi-label classifier"""

    def __init__(self, classifier=None, model_count=None, labelset_size=None, require_dense=None):
        super(RakelO, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self.model_count = int(model_count)
        self.labelset_size = labelset_size
        self.copyable_attrs = ['model_count',
                               'labelset_size', 'require_dense', 'classifier']

    def generate_partition(self, X, y):
        """Randomly divide the label space

        This function randomly divides the label space of :code:`n_labels`
        into :code:`model_count` equal subspaces of size
        :code:`labelset_size`. Sets :code:`self.partition`
        and :code:`self.label_count`.

        Parameters
        -----------
        X : numpy.ndarray or scipy.sparse
            not used, maintained for API compatibility
        y : numpy.ndarray or scipy.sparse
            binary indicator matrix with label assigments of shape
            :code:`(n_samples, n_labels)`
        """
        label_sets = []
        self.label_count = y.shape[1]
        free_labels = range(self.label_count)

        while len(label_sets) < self.model_count:
            label_set = random.sample(free_labels, self.labelset_size)
            if label_set not in label_sets:
                label_sets.append(label_set)

        self.partition = label_sets
        assert len(self.partition) == self.model_count

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

        return self.ensure_input_format(votes, enforce_sparse=False)
