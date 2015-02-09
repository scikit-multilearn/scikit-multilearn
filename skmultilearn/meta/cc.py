from ..base import MLClassifierBase
import copy
import numpy as np

class ClassifierChains(MLClassifierBase):
    """Classifier Chains multi-label classifier."""
    BRIEFNAME = "CC"
    
    def __init__(self, classifier = None):
        super(ClassifierChains, self).__init__(classifier)

    def fit(self, X, y):
        self.predictions = y;
        self.num_instances = len(y)
        self.num_labels = len(y[0])
        self.classifiers = []
        for label in xrange(self.num_labels):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, label)
            X_extended = []
            for i in xrange(self.num_instances):
                x = np.copy(X[i])
                X_extended.append(np.append(x, y[i][:label]))
            classifier.fit(X_extended, y_subset)
            self.classifiers.append(classifier)

    def predict(self, X):
        result = np.zeros((len(X), self.num_labels), dtype='i8')
        for instance in xrange(len(X)):
            predictions = []
            for label in xrange(self.num_labels):
                prediction = self.classifiers[label].predict(np.append(X[instance], predictions))
                predictions.append(prediction)
                result[instance][label] = prediction
        return result