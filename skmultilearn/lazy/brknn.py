# http://lpis.csd.auth.gr/publications/spyromitros-setn08.pdf
# come with a base kNN classifier
from ..base import MLClassifierBase
from ..utils import get_matrix_in_format
from sklearn.neighbors import NearestNeighbors
import numpy as np

class BinaryRelevanceKNN(MLClassifierBase):
    """Binary Relevance multi-label classifier based on k Nearest Neighbours method."""
    BRIEFNAME = "BRkNN"

    EXTENSION_A = 'BRkNN-a'
    EXTENSION_B = 'BRkNN-b'

    def __init__(self, k = 10, extension = None):
        # initialize the classifier by using one of BRkNN versions
        super(BinaryRelevanceKNN, self).__init__(self.classifiers(extension))
        self.BRIEFNAME = extension
        self.k = k # Number of neighbours

    def classifiers(self, extension):
        return {
            BinaryRelevanceKNN.EXTENSION_A: BRkNNaClassifier,
            BinaryRelevanceKNN.EXTENSION_B: BRkNNbClassifier
        }.get(extension, BRkNNaClassifier)

    def fit(self, X, y):
        X = self.ensure_input_format(X, sparse_format = 'csr', enforce_sparse = True)
        y = self.ensure_output_format(y, sparse_format = 'csr', enforce_sparse = True)

        self.y_ = y
        self.num_instances = y.shape[0]
        self.num_labels = y.shape[1]
        self.knn = NearestNeighbors(self.k).fit(X)
        return self

    def predict(self, X):
        X = self.ensure_input_format(X, sparse_format = 'csr', enforce_sparse = True)
        neighbors = self.knn.kneighbors(X, self.k, return_distance=False)
        classifier = self.classifier(self.k, self.num_labels)
        return classifier.fit_predict(neighbors, self.y_)       


class BaseBRkNNClassifier(object):

    def __init__(self, k, num_labels):
        self.k = k
        self.num_labels = num_labels

    def compute_confidences(self, neighbors, y):
        self.confidences = sparse.lil_matrix((y.shape[0], y.shape[1]), dtype = float)
        assert len(neighbors) == y.shape[0]
        assert self.num_labels == y.shape[1]

        for row in xrange(y.shape[0]):
            for label in xrange(y.shape[1]):
                confidence = sum(y[neighbor][label] == 1 for neighbor in neighbors[row])
                if confidence > 0:
                    self.confidences[label] = float(confidence) / self.k
            

    def fit_predict(self, neighbors, y):
        raise NotImplementedError("BaseBRkNNClassifier::fit_predict()")

class BRkNNaClassifier(BaseBRkNNClassifier):

    def fit_predict(self, neighbors, y):
        self.compute_confidences(neighbors, y)
        prediction = get_matrix_in_format(self.confidences, 'csr').rint()
        return prediction


class BRkNNbClassifier(BaseBRkNNClassifier):

    def fit_predict(self, neighbors, y):
        self.compute_confidences(neighbors, y)
        self.labels_counts_ = [sum(y[neighbor]) for neighbor in neighbors[0]]
        self.avg_labels = int(round(float(sum(labels_counts)) / len(labels_counts)))

        prediction = np.zeros(len(self.confidences), dtype='i8')
        labels_sorted = sorted(range(len(self.confidences)), key=lambda k: self.confidences[k], reverse=True)
        for label in labels_sorted[:self.avg_labels:]:
            prediction[label] = 1
        return prediction