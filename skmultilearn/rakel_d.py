from ..base import MLClassifierBase
import copy
import numpy as np
import random


class RakelD(object):
    """docstring for RakelD"""

    def __init__(self, classifier = None, labelset_size = None):
        super(RakelD, self).__init__(classifier)
        self.labelset_size = labelset_size

    def sample_models(self):
        label_sets = []
        free_labels = xrange(self.label_count)
        self.model_count = np.ceil(self.label_count/self.labelset_size)

        while len(label_sets) <= self.model_count:
            if len(free_labels) < self.labelset_size:
                label_sets.append(free_labels)
                continue

            label_set = random.sample(free_labels, self.labelset_size)
            free_labels = list(set(free_labels).difference(set(label_set)))
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

    def predict(self, X):
        results = np.zeros((len(X), self.label_count), dtype=numpy.int)
        for i in xrange(self.model_count):
            label_set = np.array(self.label_sets[i])
            for y in self.classifiers[i].predict(X):
                


        row_results = []
        for row in len(X):
            row_result = [results[i][row] for i in xrange(self.model_count)]
            row_results.append(row_result)

        return row_results


