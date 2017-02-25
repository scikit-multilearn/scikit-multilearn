from builtins import range
from ..base.problem_transformation import ProblemTransformationBase
from scipy.sparse import hstack, coo_matrix
from sklearn.utils import check_array
import copy


class BinaryRelevance(ProblemTransformationBase):

    """Binary Relevance Multi-Label Classifier.

    Transforms a multi-label classification problem with L labels
    into L single-label separate binary classification problems
    using the same base classifier provided in the constructor. The
    prediction output is the union of all per label classifiers.

    :param classifier: clonable scikit-compatible base classifier
    :type classifier: :py:class:`sklearn.base.BaseEstimator` or compatible

    :param require_dense: whether the base classifier requires dense
        representations for input features and classes/labels matrices in fit/predict.
    :type require_dense: [bool, bool]

    """

    BRIEFNAME = "BR"

    def __init__(self, classifier=None, require_dense=None):
        super(BinaryRelevance, self).__init__(classifier, require_dense)

    def generate_partition(self, X, y):
        """ Partitions the label space into singletons

            :param X: not used
            :param y: binary indicator matrix with label assignments -
                only used for learning # of labels
            :type y: matrix or sparse matrix

            Sets self.partition (list of single item lists) and self.model_count (equal to number of labels)

        """
        self.partition = list(range(y.shape[1]))
        self.model_count = y.shape[1]

    def fit(self, X, y):
        """Fit classifier with training data

        Internally this method uses a sparse CSR representation for X
        (:py:class:`scipy.sparse.csr_matrix`) and sparse CSC representation for y
        (:py:class:`scipy.sparse.csc_matrix`).

        :param X: input features
        :type X: dense or sparse matrix (n_samples, n_features)
        :param y: binary indicator matrix with label assignments
        :type y: dense or sparse matrix of {0, 1} (n_samples, n_labels)
        :returns: Fitted instance of self

        """
        X = self.ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self.ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.generate_partition(X, y)
        self.classifiers = []

        for i in range(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, self.partition[i], axis=1)
            classifier.fit(self.ensure_input_format(
                X), self.ensure_output_format(y_subset))
            self.classifiers.append(classifier)

        return self

    def predict(self, X):
        """Predict labels for X

        Internally this method uses a sparse CSR representation for X
        (:py:class:`scipy.sparse.coo_matrix`).

        :param X: input features
        :type X: dense or sparse matrix (n_samples, n_features)
        :returns: binary indicator matrix with label assignments
        :rtype: sparse matrix of int (n_samples, n_labels)

        """
        predictions = [self.ensure_multi_label_from_single_class(
            self.classifiers[label].predict(self.ensure_input_format(X)))
            for label in range(self.model_count)]

        return hstack(predictions)

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Internally this method uses a sparse CSR representation for X
        (:py:class:`scipy.sparse.coo_matrix`).

        :param X: input features
        :type X: dense or sparse matrix (n_samples, n_labels)
        :returns: matrix with label assignment probabilities
        :rtype: sparse matrix of float (n_samples, n_labels)

        """
        predictions = [self.ensure_multi_label_from_single_class(
            self.classifiers[label].predict_proba(
            self.ensure_input_format(X)))[:, 1] for label in range(self.model_count)]

        return hstack(predictions)
