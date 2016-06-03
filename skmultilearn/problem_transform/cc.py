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
        print self.label_count
        self.classifiers = [None for x in xrange(self.label_count)]

        for label in xrange(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, label, axis=1)
            print label, X_extended.shape

            self.classifiers[label] = classifier.fit(self.ensure_input_format(
                X_extended), self.ensure_output_format(y_subset))
            X_extended = hstack([X_extended, y_subset])

        return self

    def predict(self, X):
        X_extended = self.ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)
        prediction = None
        for label in xrange(self.label_count):
            print label, X_extended.shape
            prediction = self.classifiers[label].predict(
                self.ensure_input_format(X_extended))
            prediction = self.ensure_output_format(
                prediction, sparse_format='csc', enforce_sparse=True)
            X_extended = hstack([X_extended, prediction.T]).tocsc()

        return X_extended[:, -self.label_count:]
