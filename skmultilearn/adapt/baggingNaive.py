import random
import numpy, copy
from scipy import sparse
from ..base import MLClassifierBase


class BaggingNaive(MLClassifierBase):
    def __init__(self, classifier=None, require_dense=None, model_count=None):
        super(BaggingNaive, self).__init__(classifier=classifier, require_dense=require_dense)
        self.model_count=model_count

    def generate_partition(self, X, y):
        """Naive method for splitting data"""
        self.label_count = y.shape[1]

        instances_sets = []
        self.instance_count = y.shape[0]
        self.partition_size = int(numpy.ceil(self.instance_count / self.model_count))
        free_instances = xrange(self.instance_count)
        while (len(instances_sets) < self.model_count):
            instances_set = random.sample(free_instances, self.partition_size)
            free_instances = list(set(free_instances).difference(set(instances_set)))
            instances_sets.append(instances_set)

        self.partition = instances_sets

    def fit(self, X, y):

        X = self.ensure_input_format(X, sparse_format='csr', enforce_sparse=True)
        y = self.ensure_output_format(y, sparse_format='csc', enforce_sparse=True)

        self.generate_partition(X, y)

        self.classifiers = []

        for i in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            x_subset = self.generate_data_subset(X, self.partition[i], axis=0)
            y_subset = self.generate_data_subset(y, self.partition[i], axis=0)
            classifier.fit(self.ensure_input_format(x_subset), self.ensure_output_format(y_subset))
            self.classifiers.append(classifier)

        return self

    # TODO refactor to use sparse matrixes everywhere
    def predict(self, X):
        predictions = []
        for i in xrange(self.model_count):
            predictions.append(self.classifiers[i].predict(self.ensure_input_format(X)))

        result = sparse.lil_matrix((X.shape[0], self.label_count), dtype=int)

        for instance in xrange(self.instance_count / 2):
            for label in xrange(self.label_count):
                votes = []
                for model in xrange(self.model_count):
                    votes.append(predictions[model][instance][label])
                assignLabel = self.vote(votes)
                result[instance, label] = assignLabel

        return result

    def vote(self, votes):
        assignLabel = 0
        suma = sum(votes)
        if suma > (self.model_count / 2):
            assignLabel = 1
        return assignLabel
