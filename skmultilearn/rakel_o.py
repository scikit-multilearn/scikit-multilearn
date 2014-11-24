from ..base import MLClassifierBase
import copy
import numpy as np
import random

class RakelO(MLClassifierBase):
    """docstring for RakelO"""

    def __init__(self, classifier = None, models = None, labelset_size = None):
        super(RakelO, self).__init__(classifier)
        self.models = np.min(models)
        self.labelset_size = labelset_size

    def sample_models(self):
        label_sets = []
        free_labels = xrange(self.label_count)
        self.model_count = np.ceil(self.label_count/self.labelset_size)

        while len(label_sets) <= self.model_count:
            label_set = random.sample(free_labels, self.labelset_size)
            if label_set not in label_sets:
                label_sets.append(label_set)

        self.label_sets = label_sets

    def generate_data_subset(self, y, labels):
        return [row[labels] for row in y]


    def fit(self, X, y):
        self.classifiers = []
        self.label_count = len(y[0])
        self.sample_models()
        for i in xrange(self.model_count):
            classifier = copy.copy(self.classifier)
            y_subset = self.subset(y,self.label_sets[i])
            classifier.fit(X,y_subset)
            self.classifiers.append(classifier)

        return self

    def handle_voting(self, input_result):
        result = []
        sums  = [0.0 for i in xrange(self.label_count)]
        votes = [0.0 for i in xrange(self.label_count)]
        for i in xrange(self.model_count):
            for label in input_result[i]:
                sums[label] += 1.0

            for label in self.label_sets[i]:
                votes[label] += 1.0

        for i in xrange(self.label_count):
            if sums[i]/votes[i] > 0.5:
                result.append(i)

    def predict(self, X):
        results = []
        for i in xrange(self.model_count):
            label_set = np.array(self.label_sets[i])
            result = [label_set[assigned_labels] for assigned_labels in self.classifiers[i].predict(X)]
            results.append(result)

        row_results = []
        for row in len(X):
            row_result = [results[i][row] for i in xrange(self.model_count)]
            row_results.append(row_result)

        return row_results
