import copy
import numpy as np

from scipy.sparse import hstack, issparse, lil_matrix

from ..base.problem_transformation import ProblemTransformationBase
from ..base.base import MLClassifierBase


class BinaryRelevance(ProblemTransformationBase):
    """Performs classification per label

    Transforms a multi-label classification problem with L labels
    into L single-label separate binary classification problems
    using the same base classifier provided in the constructor. The
    prediction output is the union of all per label classifiers

    Parameters
    ----------
    classifier : :class:`~sklearn.base.BaseEstimator`
        scikit-learn compatible base classifier
    require_dense : [bool, bool], optional
        whether the base classifier requires dense representations
        for input features and classes/labels matrices in fit/predict.
        If value not provided, sparse representations are used if base classifier is
        an instance of :class:`~skmultilearn.base.MLClassifierBase` and dense otherwise.

    Attributes
    ----------
    model_count_ : int
        number of trained models, in this classifier equal to `n_labels`
    partition_ : List[List[int]], shape=(`model_count_`,)
        list of lists of label indexes, used to index the output space matrix, set in :meth:`_generate_partition`
        via :meth:`fit`
    classifiers_ : List[:class:`~sklearn.base.BaseEstimator`] of shape `model_count`
        list of classifiers trained per partition, set in :meth:`fit`

    Notes
    -----
    .. note ::

        This is one of the most basic approaches to multi-label classification, it ignores relationships between labels.

    Examples
    --------
    An example use case for Binary Relevance classification
    with an :class:`sklearn.svm.SVC` base classifier which supports sparse input:


    .. code-block:: python

        from skmultilearn.problem_transform import BinaryRelevance
        from sklearn.svm import SVC

        # initialize Binary Relevance multi-label classifier
        # with an SVM classifier
        # SVM in scikit only supports the X matrix in sparse representation

        classifier = BinaryRelevance(
            classifier = SVC(),
            require_dense = [False, True]
        )

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)

    Another way to use this classifier is to select the best scenario from a set of single-label classifiers used
    with Binary Relevance, this can be done using cross validation grid search. In the example below, the model
    with highest accuracy results is selected from either a :class:`sklearn.naive_bayes.MultinomialNB` or
    :class:`sklearn.svm.SVC` base classifier, alongside with best parameters for that base classifier.

    .. code-block:: python

        from skmultilearn.problem_transform import BinaryRelevance
        from sklearn.model_selection import GridSearchCV
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import SVC

        parameters = [
            {
                'classifier': [MultinomialNB()],
                'classifier__alpha': [0.7, 1.0],
            },
            {
                'classifier': [SVC()],
                'classifier__kernel': ['rbf', 'linear'],
            },
        ]


        clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
        clf.fit(x, y)

        print (clf.best_params_, clf.best_score_)

        # result:
        #
        # {
        #   'classifier': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
        #   max_iter=-1, probability=False, random_state=None, shrinking=True,
        #   tol=0.001, verbose=False), 'classifier__kernel': 'linear'
        # } 0.17

    """

    def __init__(self, classifier=None, require_dense=None):
        super(BinaryRelevance, self).__init__(classifier, require_dense)

    def _generate_partition(self, X, y):
        """Partitions the label space into singletons

        Sets `self.partition_` (list of single item lists) and `self.model_count_` (equal to number of labels).

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            not used, only for API compatibility
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `int`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        self.partition_ = list(range(y.shape[1]))
        self.model_count_ = y.shape[1]

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

        Notes
        -----
        .. note :: Input matrices are converted to sparse format internally if a numpy representation is passed
        """
        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self._ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.classifiers_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]

        for i in range(self.model_count_):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            classifier.fit(self._ensure_input_format(
                X), self._ensure_output_format(y_subset))
            self.classifiers_.append(classifier)

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
        predictions = [self._ensure_multi_label_from_single_class(
            self.classifiers_[label].predict(self._ensure_input_format(X)))
            for label in range(self.model_count_)]

        return hstack(predictions)

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

        result = lil_matrix((X.shape[0], self._label_count), dtype='float')
        for label_assignment, classifier in zip(self.partition_, self.classifiers_):
            if isinstance(self.classifier, MLClassifierBase):
                # the multilabel classifier should provide a (n_samples, n_labels) matrix
                # we just need to reorder it column wise
                result[:, label_assignment] = classifier.predict_proba(X)
            else:
                # a base classifier for binary relevance returns
                # n_samples x n_classes, where n_classes = [0, 1] - 1 is the probability of
                # the label being assigned
                result[:, label_assignment] = self._ensure_multi_label_from_single_class(
                    classifier.predict_proba(
                        self._ensure_input_format(X))
                )[:, 1]  # probability that label is assigned

        return result
