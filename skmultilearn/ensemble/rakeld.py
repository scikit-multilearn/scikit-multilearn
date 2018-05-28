import numpy as np

from .partition import LabelSpacePartitioningClassifier
from ..cluster.random import RandomLabelSpaceClusterer
from ..problem_transform import LabelPowerset


class RakelD(LabelSpacePartitioningClassifier):
    """Distinct RAndom k-labELsets multi-label classifier."""

    def __init__(self, base_classifier=None, labelset_size=None, base_classifier_require_dense=None):
        super(RakelD, self).__init__(
            classifier=LabelPowerset(classifier=base_classifier, require_dense=base_classifier_require_dense),
            clusterer=RandomLabelSpaceClusterer(
                partition_size=labelset_size,
                partition_count=None,
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
        self.clusterer.partition_count = self.model_count
        return super(RakelD, self).fit(X, y)
