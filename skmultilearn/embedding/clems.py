import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from ._mdsw import _MDSW
from copy import copy

class CLEMS:
    """Embed the label space using a label network embedder from OpenNE

    Parameters
    ----------
    measure: Callable
        a cost function executed on two label vectors
    dimension: int
        the dimension of the label embedding vectors
    is_score: boolean
        set to True if measures is a score function (higher value is better), False if loss function (lower is better)
    param_dict
        parameters passed to the embedder, don't use the dimension and graph parameters, this class will set them at fit

    Example code for using this embedder looks like this:

    .. code-block:: python

        from skmultilearn.embedding import CLEMS, EmbeddingClassifier
        from sklearn.ensemble import RandomForestRegressor
        from skmultilearn.adapt import MLkNN
        from sklearn.metrics import accuracy_score


        clf = EmbeddingClassifier(
            CLEMS(accuracy_score, 4, True),
            RandomForestRegressor(n_estimators=10),
            MLkNN(k=5),
            True
        )

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
    """
    def __init__(self, measure, dimension, is_score=False, params=None):
        self.measure = measure
        if is_score:
            self.measure = lambda x,y : 1 - measure(x,y)

        self.dimension = dimension
        if params is None:
            params = {}

        self.params = params

    def fit(self, X, y):
        """Fits the embedder to data

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        self
            fitted instance of self
        """

        self.n_labels_ = y.shape[1]

        # get unique label combinations
        if sp.issparse(y):
            _, idx = np.unique(y.rows, return_index=True)
        else:
            _, idx = np.unique(y, return_index=True)

        y_unique = y[:, idx]

        self.knn_ = NearestNeighbors(n_neighbors=1)
        self.knn_.fit(y_unique)

        nearest_points = self.knn_.kneighbors(y)[1][:, 0]
        nearest_points_counts = np.unique(nearest_points, return_counts=True)[1]

        # calculate delta matrix
        delta = np.zeros((2 * y_unique.shape[0], 2 * y_unique.shape[0]))
        for i in range(y_unique.shape[0]):
            for j in range(y_unique.shape[0]):
                delta[i, y_unique.shape[0] + j] = np.sqrt(self.measure(y_unique[None, i], y_unique[None, j]))
                delta[y_unique.shape[0] + j, i] = delta[i, y_unique.shape[0] + j]

        # calculate MDS embedding
        params = copy(self.params)
        params['n_components'] = self.dimension
        params['n_uq'] = y_unique.shape[0]
        params['uq_weight'] = nearest_points_counts
        self.embedder_ = _MDSW(**params)
        self.embedder_.fit(delta)

    def fit_transform(self,X, y):
        """Fit the embedder and transform the output space

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        X, y_embedded
            results of the embedding, input and output space
        """
        self.fit(X,y)
        return self.embedder_.embedding_
