from builtins import range
from ..base import MLClassifierBase
from ..utils import get_matrix_in_format
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import numpy as np

class BinaryRelevanceKNN(MLClassifierBase):
    """Binary Relevance multi-label classifier based on k Nearest Neighbors method."""

    def __init__(self, k = 10):
        super(BinaryRelevanceKNN, self).__init__()
        self.k = k # Number of neighbours
        self.copyable_attrs = ['k']

    def fit(self, X, y):
        self.train_labelspace = get_matrix_in_format(y, 'csc')
        self.num_instances = self.train_labelspace.shape[0]
        self.num_labels = self.train_labelspace.shape[1]
        self.knn = NearestNeighbors(self.k).fit(X)
        return self

    def compute_confidences(self):
        # % of neighbours that have a given label assigned
        # sum over each label columns after subsetting for neighbours
        # and normalize
        self.confidences = np.vstack([self.train_labelspace[n,:].tocsc().sum(axis=0) / float(self.num_labels) for n in self.neighbors])
        return self.confidences

    def predict(self, X):
        self.neighbors = self.knn.kneighbors(X, self.k, return_distance=False)
        self.compute_confidences()
        return self.predict_variant(X)

class BRkNNaClassifier(BinaryRelevanceKNN):
    """Binary Relevance multi-label classifier based on k Nearest Neighbours method.

    This version of the classifier assigns the labels that are assigned to at least half of the neighbors.
    """

    def predict_variant(self, X):
        # TODO: find out if moving the sparsity to compute confidences boots speed
        return sparse.csr_matrix(np.rint(self.confidences), dtype='i8')

class BRkNNbClassifier(BinaryRelevanceKNN):
    """Binary Relevance multi-label classifier based on k Nearest Neighbours method.

    This version of the classifier assigns the most popular m labels of the neighbors, where m is the average number of labels assigned to the object's neighbors.
    """

    def predict_variant(self, X):
        self.avg_labels = [int(np.average(self.train_labelspace[n,:].sum(axis=1)).round()) for n in self.neighbors]
        
        prediction = sparse.lil_matrix((X.shape[0], self.num_labels), dtype='i8')
        top_labels = np.argpartition(self.confidences, kth=self.avg_labels, axis=1).tolist()
        
        for i in range(X.shape[0]):
            for j in top_labels[i][-self.avg_labels[i]:]:
                prediction[i,j] += 1
        
        return prediction