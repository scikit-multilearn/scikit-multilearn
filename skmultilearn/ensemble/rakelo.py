from .voting import MajorityVotingClassifier
from ..cluster.random import RandomLabelSpaceClusterer
from ..problem_transform import LabelPowerset
from ..base import MLClassifierBase


class RakelO(MLClassifierBase):
    """Overlapping RAndom k-labELsets multi-label classifier

    Divides the label space in to m subsets of size k, trains a Label Powerset
    classifier for each subset and assign a label to an instance
    if more than half of all classifiers (majority) from clusters that contain the label
    assigned the label to the instance.

    Implements the RAkELd classifier from Tsoumakas et. al.'s paper:
    Random k-Labelsets for Multilabel Classification,
    https://ieeexplore.ieee.org/document/5567103/

    """

    def __init__(self, base_classifier=None, model_count=None, labelset_size=None, base_classifier_require_dense=None):
        """Initialize the classifier

        Attributes
        ----------
        base_classifier : sklearn.base
            the base classifier that will be used in a class, will be
            automatically put under :code:`self.classifier` for future
            access.
        base_classifier_require_dense : [bool, bool]
            whether the base classifier requires [input, output] matrices
            in dense representation. Will be automatically
            put under :code:`self.classifier.require_dense`.
        require_dense : [bool, bool]
            whether this classifier should work on [input, output] matrices
            in dense representation, required when performing cross-validation.
            Will be automatically put under :code:`self.require_dense`.
        labelset_size : int
            the desired size of each of the partitions, parameter k according to paper.
            Will be automatically put under :code:`self.labelset_size`.
        model_count : int
            the desired number of classifiers, parameter m according to paper.
            Will be automatically put under :code:`self.model_count`.
        """

        self.model_count = model_count
        self.labelset_size = labelset_size
        self.base_classifier = base_classifier
        self.base_classifier_require_dense = base_classifier_require_dense
        self.copyable_attrs = ['model_count', 'labelset_size',
                               'base_classifier_require_dense',
                               'base_classifier']

    def fit(self, X, y):
        """Fit classifier to multi-label data

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments, shape
            :code:`(n_samples, n_labels)`

        Returns
        -------
        fitted instance of self
        """
        self.classifier = MajorityVotingClassifier(
            classifier=LabelPowerset(
                classifier=self.base_classifier,
                require_dense=self.base_classifier_require_dense
            ),
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=self.labelset_size,
                cluster_count=self.model_count,
                allow_overlap=True
            ),
            require_dense=[False, False]
        )
        return self.classifier.fit(X, y)

    def predict(self, X):
        """Abstract method to predict labels

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

        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
