from ..base import MLClassifierBase
import copy
import numpy as np

class BinaryRelevance(MLClassifierBase):
    """Binary Relevance multi-label classifier."""
    BRIEFNAME = "BR"

    def __init__(self, classifier = None):
        super(BinaryRelevance, self).__init__(classifier)

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.classifiers = []
        self.label_count = len(y[0])

        for i in xrange(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y,i)
            classifier.fit(X,y_subset)
            self.classifiers.append(classifier)

        return self

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        result = np.zeros((len(X), self.label_count), dtype='i8')

        for label in xrange(self.label_count):
            prediction = self.classifiers[label].predict(X)

            for row in xrange(len(X)):
                result[row, label] = prediction[row]

        return np.array(result)
