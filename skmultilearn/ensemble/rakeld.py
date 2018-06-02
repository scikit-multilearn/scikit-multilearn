import numpy as np

from .partition import LabelSpacePartitioningClassifier
from ..cluster.random import RandomLabelSpaceClusterer
from ..problem_transform import LabelPowerset


class RakelD(LabelSpacePartitioningClassifier):
    """Distinct RAndom k-labELsets multi-label classifier.

    Divides the label space in to equal partitions of size k, trains a Label Powerset
    classifier per partition and predicts by summing the result of all trained classifiers.

    Implements the RAkELd classifier from Tsoumakas et. al.'s paper:
    Random k-Labelsets for Multilabel Classification,
    https://ieeexplore.ieee.org/document/5567103/
    """

    def __init__(self, base_classifier=None, labelset_size=None, base_classifier_require_dense=None):
        """Initialize the classifier

        Attributes
        ----------
        base_classifier : sklearn.base
            the base classifier that will be used in a class, will be
            automatically put under :code:`self.classifier` for future
            access.
        base_classifier_require_dense : [bool, bool]
            whether the base classifier requires [input, output] matrices
            in dense representation, will be automatically
            put under :code:`self.require_dense`
        labelset_size : int
            the desired size of each of the partitions, parameter k according to paper
        """
        super(RakelD, self).__init__(
            classifier=LabelPowerset(
                classifier=base_classifier,
                require_dense=base_classifier_require_dense
            ),
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=labelset_size,
                cluster_count=None,
                allow_overlap=False
            ),
            require_dense=[False, False]
        )
        self.labelset_size = labelset_size
        self.base_classifier = base_classifier
        self.base_classifier_require_dense = base_classifier_require_dense
        self.copyable_attrs = ['base_classifier', 'base_classifier_require_dense', 'labelset_size']

    def fit(self, X, y):
        self.label_count = y.shape[1]
        self.model_count = int(np.ceil(self.label_count / self.labelset_size))
        self.clusterer.cluster_count = self.model_count
        return super(RakelD, self).fit(X, y)
