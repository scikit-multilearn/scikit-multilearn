from ..problem_transform.br import BinaryRelevance
import copy
import numpy as np

from scipy import sparse
from ..utils import get_matrix_in_format


class LabelSpacePartitioningClassifier(BinaryRelevance):
    """Community detection base classifier

    Parameters
    ----------

    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.

    clusterer: an skmultilearn.cluster.base object that partitions the output space

    require_dense : [boolean, boolean]
        Whether the base classifier requires input as dense arrays, False by default for 

    """

    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        super(LabelSpacePartitioningClassifier, self).__init__(
            classifier, require_dense)
        self.clusterer = clusterer
        self.copyable_attrs = ['clusterer', 'classifier', 'require_dense']

    def generate_partition(self, X, y):
        self.partition = self.clusterer.fit_predict(X, y)
        self.model_count = len(self.partition)
        self.label_count = y.shape[1]

        return self

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        X = self.ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        result = sparse.lil_matrix((X.shape[0], self.label_count), dtype=int)

        for model in xrange(self.model_count):
            predictions = self.ensure_output_format(self.classifiers[model].predict(
                X), sparse_format=None, enforce_sparse=True).nonzero()
            for row, column in zip(predictions[0], predictions[1]):
                result[row, self.partition[model][column]] = 1

        return result
