from ..base import MLClassifierBase
from scipy.sparse import hstack, coo_matrix, issparse
import copy
import numpy as np
import random

class ClassifierChain(MLClassifierBase):
    """Classifier Chains multi-label classifier."""
    BRIEFNAME = "CC"
    
    def __init__(self, classifier = None, require_dense = False):
        super(ClassifierChain, self).__init__(classifier, require_dense)

    def fit(self, X, y):
        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output
        # 
        self.label_count = y.shape[1]
        self.classifiers = [None for x in xrange(self.label_count)]
        X_extended = X

        for label in xrange(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, label, axis = 1)

            if label > 0:
                X_extended = hstack([X_extended, y_subset])

            if not isinstance(classifier, MLClassifierBase):
                y_subset = self.ensure_1d(y_subset)

            if self.require_dense:
                classifier.fit(X_extended.toarray(), y_subset)
            else:
                classifier.fit(X_extended, y_subset)

            self.classifiers[label] = classifier

        return self

    def predict(self, X):
        prediction = None
        for label in xrange(self.label_count):
            if label > 0:
                X_extended = hstack([X_extended, prediction])

            if self.require_dense:
                prediction = self.classifiers[label].predict(X_extended.toarray())
                prediction = coo_matrix(prediction)
            else:
                prediction = self.classifiers[label].predict(X_extended)

            if not issparse(prediction):
                prediction = coo_matrix(prediction).T

        return prediction


