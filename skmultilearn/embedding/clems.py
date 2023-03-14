from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from copy import copy
from ._mdsw import _MDSW

import numpy as np
import scipy.sparse as sp


# inspired by implementation by Kuan-Hao Huang
# https://github.com/ej0cl6/csmlc


class CLEMS(BaseEstimator):
    """Embed the label space using a label network embedder from OpenNE

    Parameters
    ----------
    measure: Callable
        a cost function executed on two label vectors
    dimension: int
        the dimension of the label embedding vectors
    is_score: boolean
        set to True if measures is a score function (higher value is better), False if loss function (lower is better)
    param_dict: dict or None
        parameters passed to the embedder, don't use the dimension and graph parameters, this class will set them at fit


    Example code for using this embedder looks like this:

    .. code-block:: python

        from skmultilearn.embedding import CLEMS, EmbeddingClassifier
        from sklearn.ensemble import RandomForestRegressor
        from skmultilearn.adapt import MLkNN
        from sklearn.metrics import accuracy_score


        clf = EmbeddingClassifier(
            CLEMS(accuracy_score, True),
            RandomForestRegressor(n_estimators=10),
            MLkNN(k=5)
        )

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
    """

    def __init__(self, measure, is_score=False, params=None):
        self.measure = measure
        if is_score:
            self.measure = lambda x, y: 1 - measure(x, y)

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

        # get unique label combinations

        self.fit_transform(X, y)

    def fit_transform(self, X, y):
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

        if sp.issparse(y):
            idx = np.unique(y.tolil().rows, return_index=True)[1]
        else:
            idx = np.unique(y, axis=0, return_index=True)[1]

        y_unique = y[idx]
        n_unique = y_unique.shape[0]

        self.knn_ = NearestNeighbors(n_neighbors=1)
        self.knn_.fit(y_unique)

        nearest_points = self.knn_.kneighbors(y)[1][:, 0]
        nearest_points_counts = np.unique(nearest_points, return_counts=True)[1]

        # calculate delta matrix
        delta = np.zeros((2 * n_unique, 2 * n_unique))
        for i in range(n_unique):
            for j in range(n_unique):
                delta[i, n_unique + j] = np.sqrt(
                    self.measure(y_unique[None, i], y_unique[None, j])
                )
                delta[n_unique + j, i] = delta[i, n_unique + j]

        # calculate MDS embedding
        params = copy(self.params)
        params["n_components"] = y.shape[1]
        params["n_uq"] = n_unique
        params["uq_weight"] = nearest_points_counts
        params["dissimilarity"] = "precomputed"
        self.embedder_ = _MDSW(**params)

        y_unique_embedded = self.embedder_.fit(delta).embedding_
        y_unique_limited_to_before_trick = y_unique_embedded[n_unique:]

        knn_to_extend_embeddings_to_other_combinations = NearestNeighbors(n_neighbors=1)
        knn_to_extend_embeddings_to_other_combinations.fit(
            y_unique_limited_to_before_trick
        )
        neighboring_embeddings_indices = (
            knn_to_extend_embeddings_to_other_combinations.kneighbors(y)[1][:, 0]
        )

        return X, y_unique_embedded[neighboring_embeddings_indices]
