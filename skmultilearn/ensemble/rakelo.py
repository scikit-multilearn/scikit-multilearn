from .voting import MajorityVotingClassifier
from ..cluster.random import RandomLabelSpaceClusterer
from ..problem_transform import LabelPowerset


class RakelO(MajorityVotingClassifier):
    """Overlapping RAndom k-labELsets multi-label classifier

    Divides the label space in to m subsets of size k, trains a Label Powerset
    classifier for each subset and assign a label to an instance
    if more than half of all classifiers (majority) from clusters that contain the label
    assigned the label to the instance.

    Implements the RAkELd classifier from Tsoumakas et. al.'s paper:
    Random k-Labelsets for Multilabel Classification,
    https://ieeexplore.ieee.org/document/5567103/

    """

    def __init__(self, base_classifier=None, model_count=None, labelset_size=None, base_classifier_require_dense=None,
                 require_dense=None):
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
        super(RakelO, self).__init__(
            classifier=LabelPowerset(
                classifier=base_classifier,
                require_dense=base_classifier_require_dense
            ),
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=labelset_size,
                cluster_count=model_count,
                allow_overlap=True
            ),
            require_dense=require_dense
        )
        self.model_count = int(model_count)
        self.labelset_size = labelset_size
        self.base_classifier = base_classifier
        self.base_classifier_require_dense = base_classifier_require_dense
        self.copyable_attrs = ['model_count', 'require_dense', 'labelset_size', 'base_classifier_require_dense',
                               'base_classifier']
