from ..base import MLClassifierBase
import numpy as np
import copy, random
from scipy import sparse

class RandomOrderedClassifierChain(MLClassifierBase):
    """Classifier Chains multi-label classifier."""
    BRIEFNAME = "CC"
    
    def __init__(self, classifier = None):
        super(RandomOrderedClassifierChain, self).__init__(classifier)
        self.ordering = None

    def draw_ordering(self):
        self.ordering = random.sample(xrange(self.label_count), self.label_count)

    def fit(self, X, y):
        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output
        # 
        self.predictions = y
        self.num_instances = y.shape[0]
        self.label_count = y.shape[1]
        self.classifiers = [None for x in xrange(self.label_count)]
        self.draw_ordering()

        for label in xrange(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_tolearn = self.generate_data_subset(y, self.ordering[label], axis = 1)
            y_toinput = self.generate_data_subset(y, self.ordering[:label], axis = 1)

            X_extended = np.append(X, y_toinput, axis = 1)
            classifier.fit(X_extended, y_tolearn)
            self.classifiers[self.ordering[label]] = classifier

        return self

    def predict(self, X):
        result = np.zeros((X.shape[0], self.label_count), dtype='i8')
        for instance in xrange(X.shape[0]):
            predictions = []
            for label in self.ordering:
                prediction = self.classifiers[label].predict(np.append(X[instance], predictions))
                predictions.append(prediction)
                result[instance][label] = prediction
        return result

class EnsembleClassifierChains(MLClassifierBase):
    """docstring for EnsembleClassifierChains"""
    def __init__(self, 
                    classifier = None, 
                    model_count = None, 
                    training_sample_percentage = None,
                    threshold = None):
        super(EnsembleClassifierChains, self).__init__(classifier)
        self.model_count = model_count
        self.threshold   = threshold
        self.percentage  = training_sample_percentage
        self.models      = None


    def generate_partition(self, X, y):
        self.partition = range(y.shape[1])
        self.model_count = y.shape[1]

    def fit(self, X, y):
        X = self.ensure_input_format(X, sparse_format = 'csr', enforce_sparse = True)
        y = self.ensure_output_format(y, sparse_format = 'csc', enforce_sparse = True)

        self.generate_partition(X, y)
        self.models = []
        self.label_count = y.shape[1]
        for model in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            sampled_rows = self.generate_data_subset(X, self.partition[model], axis = 0)
            sampled_y = self.generate_data_subset(y, self.partition[model], axis = 0)
            classifier.fit(self.ensure_input_format(sampled_rows), self.ensure_output_format(sampled_y))
            self.models.append(classifier)
        return self

    def predict(self, X):
        results = np.zeros((X.shape[0], self.label_count), dtype='i8')
        for model in self.models:
            prediction = model.predict(X)
            for row in xrange(X.shape[0]):
                for label in xrange(self.label_count):
                    results[model.ordering[label]] += prediction[row][label]

        sums = np.zeros(self.label_count, dtype='float')
        for row in results:
            sums += row

        for row in xrange(len(X)):
            for label in xrange(self.label_count):
                results[row][label] = int(results[row][label]/float(sums[label]) > self.threshold)

        return results
        