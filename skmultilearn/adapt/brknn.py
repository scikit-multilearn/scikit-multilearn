from builtins import range
from ..base import MLClassifierBase
from ..utils import get_matrix_in_format
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import numpy as np


class _BinaryRelevanceKNN(MLClassifierBase):
    """Binary Relevance adapted kNN Multi-Label Classifier base class."""

    def __init__(self, k=10):
        super(_BinaryRelevanceKNN, self).__init__()
        self.k = k  # Number of neighbours
        self.copyable_attrs = ["k"]

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
        self
            fitted instance of self
        """
        self.train_labelspace = get_matrix_in_format(y, "csc")
        self._n_samples = self.train_labelspace.shape[0]
        self._n_labels = self.train_labelspace.shape[1]
        self.knn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

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
        self.neighbors_ = self.knn_.kneighbors(X, self.k, return_distance=False)
        self.confidences_ = np.vstack(
            [
                self.train_labelspace[n, :].tocsc().sum(axis=0) / self.k
                for n in self.neighbors_
            ]
        )
        return self._predict_variant(X)


class BRkNNaClassifier(_BinaryRelevanceKNN):
    """Binary Relevance multi-label classifier based on k-Nearest Neighbors method.

    This version of the classifier assigns the labels that are assigned
    to at least half of the neighbors.

    Parameters
    ----------
    k : int
        number of neighbours


    Attributes
    ----------
    knn_ : an instance of sklearn.NearestNeighbors
        the nearest neighbors single-label classifier used underneath
    neighbors_ : array of arrays of int, shape = (n_samples, k)
        k neighbors of each sample

    confidences_ : matrix of int, shape = (n_samples, n_labels)
        label assignment confidences


    References
    ----------

    If you use this method please cite the relevant paper:

    .. code :: bibtex

         @inproceedings{EleftheriosSpyromitros2008,
            author = {Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas},
            booktitle = {Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008)},
            title = {An Empirical Study of Lazy Multilabel Classification Algorithms},
            year = {2008},
            location = {Syros, Greece}
         }

    Examples
    --------

    Here's a very simple example of using BRkNNaClassifier with a fixed number of neighbors:

    .. code :: python

        from skmultilearn.adapt import BRkNNaClassifier

        classifier = BRkNNaClassifier(k=3)

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)


    You can also use :class:`~sklearn.model_selection.GridSearchCV` to find an optimal set of parameters:

    .. code :: python

        from skmultilearn.adapt import BRkNNaClassifier
        from sklearn.model_selection import GridSearchCV

        parameters = {'k': range(1,3)}
        score = 'f1_macro'

        clf = GridSearchCV(BRkNNaClassifier(), parameters, scoring=score)
        clf.fit(X, y)

    """

    def _predict_variant(self, X):
        # TODO: find out if moving the sparsity to compute confidences_ boots speed
        return sparse.csr_matrix(np.rint(self.confidences_), dtype="i8")


class BRkNNbClassifier(_BinaryRelevanceKNN):
    """Binary Relevance multi-label classifier based on k-Nearest Neighbors method.

    This version of the classifier assigns the most popular m labels of
    the neighbors, where m is the  average number of labels assigned to
    the object's neighbors.

    Parameters
    ----------
    k : int
        number of neighbours

    Attributes
    ----------
    knn_ : an instance of sklearn.NearestNeighbors
        the nearest neighbors single-label classifier used underneath
    neighbors_ : array of arrays of int, shape = (n_samples, k)
        k neighbors of each sample

    confidences_ : matrix of int, shape = (n_samples, n_labels)
        label assignment confidences


    References
    ----------

    If you use this method please cite the relevant paper:

    .. code :: bibtex

         @inproceedings{EleftheriosSpyromitros2008,
            author = {Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas},
            booktitle = {Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008)},
            title = {An Empirical Study of Lazy Multilabel Classification Algorithms},
            year = {2008},
            location = {Syros, Greece}
         }

    Examples
    --------

    Here's a very simple example of using BRkNNbClassifier with a fixed number of neighbors:

    .. code :: python

        from skmultilearn.adapt import BRkNNbClassifier

        classifier = BRkNNbClassifier(k=3)

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)


    You can also use :class:`~sklearn.model_selection.GridSearchCV` to find an optimal set of parameters:

    .. code :: python

        from skmultilearn.adapt import BRkNNbClassifier
        from sklearn.model_selection import GridSearchCV

        parameters = {'k': range(1,3)}
        score = 'f1-macro

        clf = GridSearchCV(BRkNNbClassifier(), parameters, scoring=score)
        clf.fit(X, y)

    """

    def _predict_variant(self, X):
        avg_labels = [
            int(np.average(self.train_labelspace[n, :].sum(axis=1)).round())
            for n in self.neighbors_
        ]

        prediction = sparse.lil_matrix((X.shape[0], self._n_labels), dtype="i8")
        top_labels = np.argpartition(
            self.confidences_, kth=min(avg_labels + [len(self.confidences_[0])]), axis=1
        ).tolist()

        for i in range(X.shape[0]):
            for j in top_labels[i][-avg_labels[i] :]:
                prediction[i, j] += 1

        return prediction
