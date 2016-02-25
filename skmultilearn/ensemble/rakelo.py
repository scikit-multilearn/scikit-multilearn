from ..base import MLClassifierBase
import copy
import numpy as np
import random

class RakelO(MLClassifierBase):
    """

    Overlapping RAndom k-labELsets multi-label classifier.

    """

    def __init__(self, classifier = None, models = None, labelset_size = None):
        super(RakelO, self).__init__(classifier)
        self.model_count = int(models)
        self.labelset_size = labelset_size

    def sample_models(self):
        """Internal method for sampling k-labELsets"""
        label_sets = []
        free_labels = xrange(self.label_count)
        
        while len(label_sets) < self.model_count:
            label_set = random.sample(free_labels, self.labelset_size)
            if label_set not in label_sets:
                label_sets.append(label_set)

        self.label_sets = label_sets

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.classifiers = []
        self.label_count = len(y[0])
        self.sample_models()
        for i in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y,self.label_sets[i])
            classifier.fit(X,y_subset)
            self.classifiers.append(classifier)

        return self

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        predictions = [c.predict(X) for c in self.classifiers]
        votes = np.zeros((len(X),self.label_count), dtype='i8')
        for row in xrange(len(X)):
            for model in xrange(self.model_count):
                for label in xrange(len(self.label_sets[model])):
                    votes[row,self.label_sets[model][label]] += predictions[model][row][label]

        voters = np.zeros(self.label_count, dtype='i8')
        for label_set in self.label_sets:
            voters[label_set] += 1

        for row in xrange(len(X)):
            for label in xrange(self.label_count):
                votes[row, label] = int(float(votes[row, label]) / float(voters[label]) > 0.5)

        return votes
