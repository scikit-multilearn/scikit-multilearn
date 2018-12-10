from __future__ import absolute_import

from sklearn.base import BaseEstimator


class SKLearnEmbedder(BaseEstimator):
    """Embed the label space using a scikit-compatible matrix-based embedder

    Parameters
    ----------
    embedder : sklearn.base.BaseEstimator
        a clonable instance of a scikit-compatible embedder, will be automatically
        put under :code:`self.embedder`, see .
    pass_input_space : bool (default is False)
        whether to take :code:`X` into consideration upon clustering,
        use only if you know that the embedder can handle two
        parameters for clustering, will be automatically
        put under :code:`self.pass_input_space`.


    Example code for using this embedder looks like this:

    .. code-block:: python

        from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
        from sklearn.manifold import SpectralEmbedding
        from sklearn.ensemble import RandomForestRegressor
        from skmultilearn.adapt import MLkNN

        clf = EmbeddingClassifier(
            SKLearnEmbedder(SpectralEmbedding(n_components = 10)),
            RandomForestRegressor(n_estimators=10),
            MLkNN(k=5)
        )

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
    """

    def __init__(self, embedder=None, pass_input_space=False):
        super(BaseEstimator, self).__init__()

        self.embedder = embedder
        self.pass_input_space = pass_input_space

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

        self.embedder.fit(X, y)

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

        if self.pass_input_space:
            result = self.embedder.fit_transform(X, y)
        else:
            result = self.embedder.fit_transform(y)

        return X, result
