from skmultilearn.base import ProblemTransformationBase
import numpy as np
import scipy.sparse as sp
from copy import copy


class EmbeddingClassifier(ProblemTransformationBase):
    """Embedding-based classifier

    Implements a general scheme presented in LNEMLC: label network embeddings for multi-label classification. The
    classifier embeds the label space with the embedder, trains a set of single-variate or a multi-variate regressor
    for embedding unseen cases and a base classifier to predict labels based on input features and the embeddings.

    Parameters
    ----------
    embedder : :class:`~sklearn.base.BaseEstimator`
        the class to embed the label space

    regressor : :class:`~sklearn.base.BaseEstimator`
        the base regressor to predict embeddings from input features

    classifier : :class:`~sklearn.base.BaseEstimator`
        the base classifier to predict labels from input features and embeddings

    regressor_per_dimension : bool
        whether to train one joint multi-variate regressor (False) or per dimension single-variate regressor (True)

    require_dense : [bool, bool], optional
        whether the base classifier requires dense representations for input features and classes/labels
        matrices in fit/predict.


    Attributes
    ----------
    n_regressors_ : int
        number of trained regressors
    partition_ : List[List[int]], shape=(`model_count_`,)
        list of lists of label indexes, used to index the output space matrix, set in :meth:`_generate_partition`
        via :meth:`fit`
    classifiers_ : List[:class:`~sklearn.base.BaseEstimator`] of shape `model_count`
        list of classifiers trained per partition, set in :meth:`fit`


    If you use this classifier please cite the relevant embedding method paper
    and the label network embedding for multi-label classification paper:

    .. code :: bibtex

        @article{zhang2007ml,
          title={ML-KNN: A lazy learning approach to multi-label learning},
          author={Zhang, Min-Ling and Zhou, Zhi-Hua},
          journal={Pattern recognition},
          volume={40},
          number={7},
          pages={2038--2048},
          year={2007},
          publisher={Elsevier}
        }

    Example
    -------
    An example use case for EmbeddingClassifier:

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

    def __init__(
        self,
        embedder,
        regressor,
        classifier,
        regressor_per_dimension=False,
        require_dense=None,
    ):
        super(EmbeddingClassifier, self).__init__()
        self.embedder = embedder
        self.regressor = regressor
        self.classifier = classifier
        self.regressor_per_dimension = regressor_per_dimension

        if require_dense is None:
            require_dense = [True, True]

        self.require_dense = require_dense

        self.copyable_attrs = [
            "embedder",
            "regressor",
            "classifier",
            "regressor_per_dimension",
            "require_dense",
        ]

    def fit(self, X, y):
        """Fits classifier to training data

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

        X = self._ensure_input_format(X)
        y = self._ensure_input_format(y)

        y_embedded = self.embedder.fit_transform(X, y)[1]
        X_y_embedded = self._concatenate_matrices(X, y_embedded)

        if self.regressor_per_dimension:
            self.n_regressors_ = y_embedded.shape[1]
            self.regressors_ = [None for _ in range(self.n_regressors_)]
            for i in range(self.n_regressors_):
                self.regressors_[i] = copy(self.regressor)
                self.regressors_[i].fit(X, y_embedded[:, i])
        else:
            self.n_regressors_ = 1

            self.regressor.fit(X, y_embedded)
        self.classifier.fit(X_y_embedded, y)

        return self

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """

        X = self._ensure_input_format(X)

        X_y_embedded = self._predict_embedding(X)
        return self.classifier.predict(X_y_embedded)

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `float in [0.0, 1.0]`, shape=(n_samples, n_labels)
            matrix with label assignment probabilities
        """

        X_y_embedded = self._predict_embedding(X)
        return self.classifier.predict_proba(X_y_embedded)

    def _concatenate_matrices(self, X, y_embedded):
        X = self._ensure_input_format(X)
        y_embedded = self._ensure_input_format(y_embedded)

        if sp.issparse(X):
            X_y_embedded = sp.hstack([X, y_embedded])
        else:
            X_y_embedded = np.hstack([X, y_embedded])

        return X_y_embedded

    def _predict_embedding(self, X):
        if self.regressor_per_dimension:
            y_embedded = [
                self.regressors_[i].predict(X) for i in range(self.n_regressors_)
            ]
            if sp.issparse(X):
                y_embedded = sp.csr_matrix(y_embedded).T
            else:
                y_embedded = np.matrix(y_embedded).T
        else:
            y_embedded = self.regressor.predict(X)

        return self._concatenate_matrices(X, y_embedded)
