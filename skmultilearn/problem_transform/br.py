import copy
from builtins import range

import numpy as np
from scipy.sparse import hstack, issparse

from ..base.problem_transformation import ProblemTransformationBase


class BinaryRelevance(ProblemTransformationBase):
    """Binary Relevance Multi-Label Classifier.

    Transforms a multi-label classification problem with L labels
    into L single-label separate binary classification problems
    using the same base classifier provided in the constructor. The
    prediction output is the union of all per label classifiers.
    """

    BRIEFNAME = "BR"

    def __init__(self, classifier=None, require_dense=None):
        """Initializes the BinaryRelevance class

        Attributes
        ----------
        classifier : sklear.base.BaseEstimator
            scikit-compatible base classifier
        require_dense : list of bools ([bool, bool])
            whether the base classifier requires dense representations
            for input features and classes/labels matrices in fit/predict.
        """
        super(BinaryRelevance, self).__init__(classifier, require_dense)

    def generate_partition(self, X, y):
        """Partitions the label space into singletons

        Sets :code:`self.partition` (list of single item lists) and
        :code:`self.model_count` (equal to number of labels).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            not used, only for API compatibility
        y : numpy.ndarray or scipy.sparse
            binary indicator matrix with label assignments. Use for
            learning the number of labels
        """
        self.partition = list(range(y.shape[1]))
        self.model_count = y.shape[1]

    def fit(self, X, y):
        """Fit classifier with training data

        Internally this method uses a sparse CSR representation for X
        (:class:`scipy.sparse.csr_matrix`) and sparse CSC representation for y
        (:class:`scipy.sparse.csc_matrix`).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        skmultilearn.problem_transform.br.BinaryRelevance
            fitted instance of self
        """
        X = self.ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self.ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.generate_partition(X, y)
        self.classifiers = []

        for i in range(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, self.partition[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            classifier.fit(self.ensure_input_format(
                X), self.ensure_output_format(y_subset))
            self.classifiers.append(classifier)

        return self

    def predict(self, X):
        """Predict labels for X

        Internally this method uses a sparse CSR representation for X
        (:class:`scipy.sparse.coo_matrix`).

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
        predictions = [self.ensure_multi_label_from_single_class(
            self.classifiers[label].predict(self.ensure_input_format(X)))
            for label in range(self.model_count)]

        return hstack(predictions)

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Internally this method uses a sparse CSR representation for X
        (:class:`scipy.sparse.coo_matrix`).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of float
            matrix with label assignment probabilities of shape
            :code:`(n_samples, n_labels)`
        """
        predictions = [self.ensure_multi_label_from_single_class(
            self.classifiers[label].predict_proba(
                self.ensure_input_format(X)))[:, 1] for label in range(self.model_count)]

        return hstack(predictions)
