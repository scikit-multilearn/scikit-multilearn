import copy
import numpy as np
from .base import MLClassifierBase
from ..utils import get_matrix_in_format, matrix_creation_function_for_format
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin


class ProblemTransformationBase(MLClassifierBase):
    """Base class providing common functions for multi-label classifiers that follow the problem transformation approach.

    Problem transformation is the approach in which the original multi-label classification problem     is transformed into one or more single-label problems, which are then solved by single-class or multi-class classifiers.

    Scikit-multilearn provides a number of such methods:

    - Binary Relevance - which performs a single-label single-class classification for each label and sums the results :class:`BinaryRelevance`
    - Classifier Chains - which performs a single-label single-class classification for each label and sums the results :class:`ClassifierChain`
    - Label Powerset - which performs a single-label single-class classification for each label and sums the results :class:`LabelPowerset`

    Parameters
    ----------

    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.
    require_dense : boolean
        Whether the base classifier requires input as dense arrays, False by default
    """

    def __init__(self, classifier=None, require_dense=None):

        super(ProblemTransformationBase, self).__init__()

        self.copyable_attrs = ["classifier", "require_dense"]

        self.classifier = classifier
        if require_dense is not None:
            if isinstance(require_dense, bool):
                self.require_dense = [require_dense, require_dense]
            else:
                assert len(require_dense) == 2 and isinstance(
                    require_dense[0], bool) and isinstance(require_dense[1], bool)
                self.require_dense = require_dense

        else:
            if isinstance(self.classifier, MLClassifierBase):
                self.require_dense = [False, False]
            else:
                self.require_dense = [True, True]
