import copy
import numpy as np

from scipy.sparse import hstack, issparse, lil_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from ..base.problem_transformation import ProblemTransformationBase
from ..base.base import MLClassifierBase


class InstanceBasedLogisticRegression(ProblemTransformationBase):
    def __init__(self, classifier=LogisticRegression(), require_dense=None):
        """Combining Instance-Based Learning and Logistic Regression

        This idea is put into practice by means of a learning algorithm
        that realizes instance-based classification as
        logistic regression, using the information coming from the neighbors
        of an instance x as a “feature”.

        The first classifier layer is filled with K-Nearest Neighbor models,
        while the second classifier layer is filled with another classifiers, Logistic Regression as
        default.

        Parameters
        ----------
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
        knn_layer_ : List[:class:`~sklearn.base.BaseEstimator`] of shape `model_count`
            list of classifiers trained per partition in first layer, set in :meth:`fit`

        References
        ----------
        If used, please cite the scikit-multilearn library and the relevant paper:

        .. code-block:: bibtex

        @article{cheng2009combining,
            title={Combining instance-based learning and logistic regression for multilabel classification},
            author={Cheng, Weiwei and H{\"u}llermeier, Eyke},
            journal={Machine Learning},
            volume={76},
            number={2-3},
            pages={211--225},
            year={2009},
            publisher={Springer}
        }

        """
        super(InstanceBasedLogisticRegression, self).__init__(classifier, require_dense)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        self.knn_layer_ = []

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

    def concatenate_class_membership(self, X, class_membership):
        """Concatenate original features and instance-based information from first layer
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        class_membership : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_labels)
            label predict probabilities from X
        """

        concatenated = lil_matrix((X.shape[0], X.shape[1] + class_membership.shape[1]), dtype='float')
        concatenated = hstack([X, class_membership])

        return concatenated

    def get_class_membership(self, classifiers, X):
        """ Extract instance-based information from original data X
        Parameters
        ----------

        classifiers_ : List[:class:`~sklearn.base.BaseEstimator`] of shape `model_count`
            list of pretrained classifiers per partition based on X
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        """

        result = lil_matrix((X.shape[0], self._label_count), dtype='float')

        for label_assignment, classifier in zip(self.partition_, classifiers):
            # n_samples x n_classes, where n_classes = [0, 1] - 1 is the probability of
            # the label being assigned
            result[:, label_assignment] = self._ensure_multi_label_from_single_class(
                classifier.predict_proba(
                    self._ensure_input_format(X))
            )[:, 1]  # probability that label is assigned

        return result

    def fit(self, X, y):
        """Fits classifiers for training with given data

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
            classifier = copy.deepcopy(self.knn_classifier)
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            classifier.fit(self._ensure_input_format(
                X), self._ensure_output_format(y_subset))
            self.classifiers_.append(classifier)

        self.knn_layer_ = copy.deepcopy(self.classifiers_)
        self.classifiers_ = []

        class_membership = self.get_class_membership(self.knn_layer_, X)
        X_concat_class_membership = self.concatenate_class_membership(X, class_membership)

        for i in range(self.model_count_):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            classifier.fit(self._ensure_input_format(X_concat_class_membership),
                           self._ensure_output_format(y_subset))
            self.classifiers_.append(classifier)

        return self

    def predict(self, X):
        """Predict labels from X
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        class_membership = self.get_class_membership(self.knn_layer_, X)
        X_test_concat_membership = self.concatenate_class_membership(X, class_membership)

        predictions = [self._ensure_multi_label_from_single_class(
            self.classifiers_[label].predict(self._ensure_input_format(X_test_concat_membership)))
            for label in range(self.model_count_)]

        return hstack(predictions)

    def predict_proba(self, X):
        """Predict probabilities of each labels from given X
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        Returns
        -------
        :mod:`scipy.sparse` matrix of `float in [0.0, 1.0]`, shape=(n_samples, n_labels)
            matrix with label assignment probabilities
        """
        class_membership = self.get_class_membership(self.knn_layer_, X)
        X_test_concat_class_membership = self.concatenate_class_membership(X, class_membership)

        result = lil_matrix((X.shape[0], self._label_count), dtype='float')

        for label_assignment, classifier in zip(self.partition_, self.classifiers_):
            if isinstance(self.classifier, MLClassifierBase):
                # the multilabel classifier should provide a (n_samples, n_labels) matrix
                # we just need to reorder it column wise
                result[:, label_assignment] = classifier.predict_proba(X_test_concat_class_membership)
            else:
                # a base classifier for binary relevance returns
                # n_samples x n_classes, where n_classes = [0, 1] - 1 is the probability of
                # the label being assigned
                result[:, label_assignment] = self._ensure_multi_label_from_single_class(
                    classifier.predict_proba(
                        self._ensure_input_format(X_test_concat_class_membership))
                )[:, 1]  # probability that label is assignedx

        return result