from .voting import MajorityVotingClassifier
from ..cluster.random import RandomLabelSpaceClusterer
from ..problem_transform import LabelPowerset

class RakelO(MajorityVotingClassifier):
    """Overlapping RAndom k-labELsets multi-label classifier"""

    def __init__(self, base_classifier=None, model_count=None, labelset_size=None, base_classifier_require_dense=None, require_dense=None):
        super(RakelO, self).__init__(
            classifier=LabelPowerset(classifier=base_classifier, require_dense=base_classifier_require_dense),
            clusterer=RandomLabelSpaceClusterer(
                partition_size=labelset_size,
                partition_count=model_count,
                allow_overlap=True),
            require_dense=require_dense)
        self.model_count = int(model_count)
        self.labelset_size = labelset_size
        self.base_classifier = base_classifier
        self.base_classifier_require_dense = base_classifier_require_dense
        self.copyable_attrs = ['model_count', 'require_dense', 'labelset_size', 'base_classifier_require_dense', 'base_classifier']
