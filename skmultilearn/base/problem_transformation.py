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

    def get_params(self, deep=True):
        """
        Introspection of classifier for search models like cross validation and grid
        search.
        Parameters
        ----------
        deep : boolean
            If true all params will be introspected also and appended to the output dict.
        Returns
        -------
        out : dictionary
            Dictionary of all parameters and their values. If deep=True the dictionary
            also holds the parameters of the parameters.
        """
        out = dict()

        for attr in self.copyable_attrs:
            out[attr] = getattr(self, attr)

            if hasattr(getattr(self, attr), 'get_params') and deep:
                deep_items = getattr(self, attr).get_params().items()
                out.update((attr + '__' + k, val) for k, val in deep_items)

        return out

    def set_params(self, **parameters):
        """
        Set parameters as returned by `get_params`.
        @see https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L243
        """

        if not parameters:
            return self

        valid_params = self.get_params(deep=True)

        for parameter, value in parameters.items():
            split = parameter.split('__', 1)

            if len(split) > 1:
                sub_obj_name, sub_param = split

                if sub_obj_name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))

                sub_object = valid_params[sub_obj_name]
                sub_object.set_params(**{sub_param: value})
            else:
                if parameter in valid_params:
                    setattr(self, parameter, value)
                else:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))

        return self
