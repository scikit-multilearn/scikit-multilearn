from builtins import zip
from builtins import range
from ..problem_transform.br import BinaryRelevance
import copy
import numpy as np

from scipy import sparse
from ..utils import get_matrix_in_format


class LabelSpacePartitioningClassifier(BinaryRelevance):
    """Community detection base classifier"""

    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        """Initializes the classifier

        Attributes
        ----------
        classifier : sklearn.base
            the base classifier that will be used in a class, will be
            automatically put under :code:`self.classifier` for future
            access.
        clusterer : sklmultilear.cluster.base
            object that partitions the output space
        require_dense : list of bools ([bool, bool])
            whether the base classifier requires input as dense arrays
        """
        super(LabelSpacePartitioningClassifier, self).__init__(
            classifier, require_dense)
        self.clusterer = clusterer
        self.copyable_attrs = ['clusterer', 'classifier', 'require_dense']

    def generate_partition(self, X, y):
        """Assign fixed partition of the label space

        Mock function, the partition is assigned in the constructor.
        It sets :code:`self.model_count` and :code:`self.label_count`.

        Parameters
        -----------
        X : numpy.ndarray or scipy.sparse
            not used, maintained for API compatibility
        y : numpy.ndarray or scipy.sparse
            binary indicator matrix with label assigments of shape
            :code:`(n_samples, n_labels)`

        Returns
        -------
        LabelSpacePartitioningClassifier
            returns an instance of itself
        """
        self.partition = self.clusterer.fit_predict(X, y)
        self.model_count = len(self.partition)
        self.label_count = y.shape[1]

        return self

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """
        X = self.ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        result = sparse.lil_matrix((X.shape[0], self.label_count), dtype=int)

        for model in range(self.model_count):
            predictions = self.ensure_output_format(self.classifiers[model].predict(
                X), sparse_format=None, enforce_sparse=True).nonzero()
            for row, column in zip(predictions[0], predictions[1]):
                result[row, self.partition[model][column]] = 1

        return result
