from skmultilearn.base import ProblemTransformationBase
import numpy as np
import scipy.sparse as sp


class EmbeddingClassifier(ProblemTransformationBase):
    """Assigns k most frequent labels

    Parameters
    ----------
    embedder : int
        number of most frequent labels to assign

    regressor : int
        number of most frequent labels to assign

    classifier : int
        number of most frequent labels to assign

    regressor_per_dimension : int
        number of most frequent labels to assign


    Example
    -------
    An example use case for EmbeddingClassifier:

    .. code-block:: python

        from skmultilearn.<YOUR_CLASSIFIER_MODULE> import AssignKBestLabels

        # initialize LabelPowerset multi-label classifier with a RandomForest
        classifier = AssignKBestLabels(
            k = 3
        )

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)
    """

    def __init__(self, embedder, regressor, classifier, regressor_per_dimension=False, require_dense=None):
        super(EmbeddingClassifier, self).__init__()
        self.embedder = embedder
        self.regressor = regressor
        self.classifier = classifier
        self.regressor_per_dimension = regressor_per_dimension

        if require_dense is None:
            require_dense = [True, True]

        self.require_dense = require_dense

        self.copyable_attrs = ['embedder', 'regressor', 'classifier', 'regressor_per_dimension', 'require_dense']

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
            self.regressors_ = [None for _ in range(self.n_dim_)]
            for i in range(self.n_dim_):
                self.regressors_[i] = self.regressor
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
        y = self._ensure_input_format(y_embedded)

        if sp.issparse(X):
            X_y_embedded = sp.hstack([X, y_embedded])
        else:
            X_y_embedded = np.hstack([X, y_embedded])

        return X_y_embedded

    def _predict_embedding(self, X):
        if self.regressor_per_dimension:
            y_embedded = sp.dok_matrix((X.shape[0], self.n_dim_))
            for i in range(self.n_dim_):
                y_embedded[:, i] = self.regressors_[i].predict(X)
        else:
            y_embedded = self.regressor.predict(X)

        return self._concatenate_matrices(X, y_embedded)
