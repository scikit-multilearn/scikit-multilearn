from builtins import range
from ..base import MLClassifierBase
from ..utils import get_matrix_in_format
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import numpy as np

class BinaryRelevanceKNN(MLClassifierBase):
    """Binary Relevance adapted kNN Multi-Label Classifier."""

    def __init__(self, k = 10):
        """Initializes the classifier

        Attributes
        ----------
        k : int (default is 10)
            number of neighbours
        """
        super(BinaryRelevanceKNN, self).__init__()
        self.k = k # Number of neighbours
        self.copyable_attrs = ['k']

    def fit(self, X, y):
        """Fit classifier with training data

        Internally this method uses a sparse CSC representation for y
        (:class:`scipy.sparse.csc_matrix`).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        skmultilearn.adapt.brknn.BinaryRelevanceKNN
            fitted instance of self
        """
        self.train_labelspace = get_matrix_in_format(y, 'csc')
        self.num_instances = self.train_labelspace.shape[0]
        self.num_labels = self.train_labelspace.shape[1]
        self.knn = NearestNeighbors(self.k).fit(X)
        return self

    def compute_confidences(self):
        """Helper function to compute for the confidences

        Performs a computation involving the percent of neighbours that
        have a given label assigned, then summed over each label columns
        after subsetting for neighbours.Then normalization is done.
        """
        self.confidences = np.vstack([self.train_labelspace[n,:].tocsc().sum(axis=0) / float(self.num_labels) for n in self.neighbors])
        return self.confidences

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """
        self.neighbors = self.knn.kneighbors(X, self.k, return_distance=False)
        self.compute_confidences()
        return self.predict_variant(X)

class BRkNNaClassifier(BinaryRelevanceKNN):
    """Binary Relevance multi-label classifier based on k-Nearest
    Neighbours method.

    This version of the classifier assigns the labels that are assigned
    to at least half of the neighbors.

    Attributes
    ----------
    k : int
        number of neighbours

    """

    def predict_variant(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        # TODO: find out if moving the sparsity to compute confidences boots speed
        return sparse.csr_matrix(np.rint(self.confidences), dtype='i8')

class BRkNNbClassifier(BinaryRelevanceKNN):
    """Binary Relevance multi-label classifier based on k-Nearest
    Neighbours method.

    This version of the classifier assigns the most popular m labels of
    the neighbors, where m is the  average number of labels assigned to
    the object's neighbors.

    Attributes
    ----------
    k : int
        number of neighbours
    """

    def predict_variant(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        self.avg_labels = [int(np.average(self.train_labelspace[n,:].sum(axis=1)).round()) for n in self.neighbors]

        prediction = sparse.lil_matrix((X.shape[0], self.num_labels), dtype='i8')
        top_labels = np.argpartition(self.confidences, kth=min(self.avg_labels, len(self.confidences[0])), axis=1).tolist()

        for i in range(X.shape[0]):
            for j in top_labels[i][-self.avg_labels[i]:]:
                prediction[i,j] += 1

        return prediction