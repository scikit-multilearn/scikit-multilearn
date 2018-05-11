from builtins import range
from ..base.problem_transformation import ProblemTransformationBase
from scipy.sparse import hstack, coo_matrix, issparse
import copy
import numpy as np
import random


class ClassifierChain(ProblemTransformationBase):
    """Classifier Chains Multi-Label Classifier.

    This class provides implementation of Jesse Read's problem
    transformation method called Classifier Chains. For L labels it
    trains L classifiers ordered in a chain according to the
    `Bayesian chain rule`_.

    The first classifier is trained just on the input space, and then
    each next classifier is trained on the input space and all previous
    classifiers in the chain.

    The default classifier chains follow the same ordering as provided
    in the training set, i.e. label in column 0, then 1, etc.

    You can find more information about this method in Jesse Read's
    `ECML presentation`_ or `journal paper`_.

    .. _Bayesian chain rule: https://en.wikipedia.org/wiki/Chain_rule_(probability)
    .. _ECML presentation: https://users.ics.aalto.fi/jesse/talks/chains-ECML-2009-presentation.pdf
    .. _journal paper: http://www.cs.waikato.ac.nz/~eibe/pubs/ccformlc.pdf

    """
    BRIEFNAME = "CC"

    def __init__(self, classifier=None, require_dense=None):
        """Initializes the ClassifierChain class

        Attributes
        ----------
        classifier : sklear.base.BaseEstimator
            scikit-compatible base classifier
        require_dense : list of bools ([bool, bool])
            whether the base classifier requires dense representations
            for input features and classes/labels matrices in fit/predict.
        """
        super(ClassifierChain, self).__init__(classifier, require_dense)

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
        skmultilearn.problem_transform.cc.ClassifierChain
            fitted instance of self
        """

        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output

        X_extended = self.ensure_input_format(X, sparse_format='csc', enforce_sparse=True)
        y = self.ensure_output_format(y, sparse_format='csc', enforce_sparse=True)

        self.label_count = y.shape[1]
        self.classifiers = [None for x in range(self.label_count)]

        for label in range(self.label_count):
            self.classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, label, axis=1)

            self.classifiers[label] = self.classifier.fit(self.ensure_input_format(
                X_extended), self.ensure_output_format(y_subset))
            X_extended = hstack([X_extended, y_subset])

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

        X_extended = self.ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)
        prediction = None
        for label in range(self.label_count):
            prediction = self.classifiers[label].predict(
                self.ensure_input_format(X_extended))
            prediction = self.ensure_multi_label_from_single_class(prediction)
            X_extended = hstack([X_extended, prediction])
        return X_extended[:, -self.label_count:]

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
        X_extended = self.ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)
        prediction = None
        results = []
        for label in range(self.label_count):
            prediction = self.classifiers[label].predict(
                self.ensure_input_format(X_extended))

            prediction = self.ensure_output_format(
                prediction, sparse_format='csc', enforce_sparse=True)

            prediction_proba = self.classifiers[label].predict_proba(
                self.ensure_input_format(X_extended))

            prediction_proba = self.ensure_output_format(
                prediction_proba, sparse_format='csc', enforce_sparse=True)[:, 1]

            X_extended = hstack([X_extended, prediction]).tocsc()
            results.append(prediction_proba)

        return hstack(results)
