from builtins import range
from ..base.problem_transformation import ProblemTransformationBase
from scipy.sparse import hstack, coo_matrix, issparse
import copy
import numpy as np
import random


class ClassifierChain(ProblemTransformationBase):
    """Classifier Chains multi-label classifier."""
    BRIEFNAME = "CC"

    def __init__(self, classifier=None, require_dense=False):
        super(ClassifierChain, self).__init__(classifier, require_dense)

    def fit(self, X, y):
        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output
        #
        X_extended = self.ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)
        y = self.ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)
        self.label_count = y.shape[1]
        self.classifiers = [None for x in range(self.label_count)]

        for label in range(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, label, axis=1)

            self.classifiers[label] = classifier.fit(self.ensure_input_format(
                X_extended), self.ensure_output_format(y_subset))
            X_extended = hstack([X_extended, y_subset])

        return self

    def predict(self, X):
        X_extended = self.ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)
        prediction = None
        for label in range(self.label_count):
            prediction = self.classifiers[label].predict(
                self.ensure_input_format(X_extended))
            prediction = self.ensure_output_format(
                prediction, sparse_format='csc', enforce_sparse=True)
            X_extended = hstack([X_extended, prediction.T]).tocsc()

        return X_extended[:, -self.label_count:]

    def predict_proba(self, X):
        """Predict probabilities for labels for `X`, see base method's documentation."""
        X_extended = self.ensure_input_format(X, sparse_format='csc', enforce_sparse=True)
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

            X_extended = hstack([X_extended, prediction.T]).tocsc()
            results.append(prediction_proba)

        return hstack(results)