from ..base import MLClassifierBase
from sklearn.neighbors import NearestNeighbors
import numpy as np
import six

class BinaryRelevanceKNN(MLClassifierBase):
    """Binary Relevance multi-label classifier based on k Nearest Neighbours method."""
    BRIEFNAME = "BRkNN"

    EXTENSION_A = 'BRkNN-a'
    EXTENSION_B = 'BRkNN-b'

    def __init__(self, k = 10, extension = None):
        super(BinaryRelevanceKNN, self).__init__(self.classifiers(extension))
        self.BRIEFNAME = extension
        self.k = k # Number of neighbours

    def classifiers(self, extension):
        return {
            BinaryRelevanceKNN.EXTENSION_A: BRkNNaClassifier,
            BinaryRelevanceKNN.EXTENSION_B: BRkNNbClassifier
        }.get(extension, BRkNNaClassifier)

    def fit(self, X, y):
        self.predictions = y;
        self.num_instances = len(y)
        self.num_labels = len(y[0])
        self.knn = NearestNeighbors(self.k).fit(X)
        return self

    def predict(self, X):
        result = np.zeros((len(X), self.num_labels), dtype='i8')
        for instance in six.moves.range(len(X)):
            neighbors = self.knn.kneighbors(X[instance], self.k, return_distance=False)
            classifier = self.classifier(self.k, self.num_labels)
            classifier.fit(neighbors, self.predictions)
            result[instance] = classifier.predict()
        return result


class BaseBRkNNClassifier(object):

    def __init__(self, k, num_labels):
        self.k = k
        self.num_labels = num_labels

    def compute_confidences(self, neighbors, y):
        self.confidences = [0] * self.num_labels
        for label in six.moves.range(self.num_labels):
            confidence = sum(y[neighbor][label] == 1 for neighbor in neighbors[0])
            self.confidences[label] = float(confidence) / self.k

    def fit(self, neighbors, y):
        raise NotImplementedError("BaseBRkNNClassifier::fit()")

    def predict(self):
        raise NotImplementedError("BaseBRkNNClassifier::predict()")

class BRkNNaClassifier(BaseBRkNNClassifier):

    def fit(self, neighbors, y):
        self.compute_confidences(neighbors, y)

    def predict(self):
        prediction = [1 if confidence >= 0.5 else 0 for confidence in self.confidences]
        return np.array(prediction)


class BRkNNbClassifier(BaseBRkNNClassifier):

    def fit(self, neighbors, y):
        self.compute_confidences(neighbors, y)
        labels_counts = [sum(y[neighbor]) for neighbor in neighbors[0]]
        self.avg_labels = int(round(float(sum(labels_counts)) / len(labels_counts)))

    def predict(self):
        prediction = np.zeros(len(self.confidences), dtype='i8')
        labels_sorted = sorted(range(len(self.confidences)), key=lambda k: self.confidences[k], reverse=True)
        for label in labels_sorted[:self.avg_labels:]:
            prediction[label] = 1
        return prediction