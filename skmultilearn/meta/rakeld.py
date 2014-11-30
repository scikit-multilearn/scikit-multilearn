from ..base import MLClassifierBase
import copy
import random
import numpy as np

class RakelD(MLClassifierBase):
    """Distinct RAndom k-labELsets multi-label classifier."""

    def __init__(self, classifier = None, labelset_size = None):
        super(RakelD, self).__init__(classifier)
        self.labelset_size = labelset_size

    def sample_models(self, label_count = None):
        if label_count is not None:
            self.label_count = label_count

        """Internal method for sampling k-labELsets"""
        label_sets = []
        free_labels = xrange(self.label_count)
        self.model_count = int(np.ceil(self.label_count/self.labelset_size))

        while len(label_sets) <= self.model_count:
            if len(free_labels) == 0:
                break
            if len(free_labels) < self.labelset_size:
                label_sets.append(free_labels)
                continue

            label_set = random.sample(free_labels, self.labelset_size)
            free_labels = list(set(free_labels).difference(set(label_set)))
            label_sets.append(label_set)

        self.label_sets = label_sets

    def fit_only(self, X, y):
        """Fit classifier according to X,y, without resampling models."""
        self.classifiers = []
        for i in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y,self.label_sets[i])
            classifier.fit(X,y_subset)
            self.classifiers.append(classifier)

        return self

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.sample_models(len(y[0]))
        return self.fit_only()

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        input_rows = len(X)
        predictions = [self.classifiers[i].predict(X) for i in xrange(self.model_count)]
        result = np.zeros((input_rows, self.label_count))

        for row in xrange(input_rows):
            for model in xrange(self.model_count):
                for label_position in xrange(len(self.label_sets[model])):
                    if predictions[model][row][label_position] == 1:
                        label_id = self.label_sets[model][label_position]
                        result[row][label_id] = 1
        return result
