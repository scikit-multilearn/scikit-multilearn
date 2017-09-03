from __future__ import absolute_import
from builtins import range
from .base import LabelSpaceClustererBase
import numpy as np


class MatrixLabelSpaceClusterer(LabelSpaceClustererBase):
    """Clusters the label space using a matrix-based clusterer"""

    def __init__(self, clusterer=None, pass_input_space=False):
        """Initializes the clusterer

        Attributes
        ----------
        clusterer : sklearn.base.ClusterMixin
            a clonable instance of a scikit-compatible clusterer
        pass_input_space : bool (default is False)
            whether to take :code:`X` into consideration upon clustering,
            use only if you know that the clusterer can handle two
            parameters for clustering
        """
        super(MatrixLabelSpaceClusterer, self).__init__()

        self.clusterer = clusterer
        self.pass_input_space = pass_input_space

    def fit_predict(self, X, y):
        """Cluster the output space

        Uses the :code:`fit_predict` method of provided :code:`clusterer`
        to perform label space division.

        Returns
        -------
        numpy.ndarray
            partition of labels, each sublist contains label indices
            related to label positions in :code:`y`
        """

        if self.pass_input_space:
            return self.clusterer.fit_predict(X, y.transpose())

        return self.clusterer.fit_predict(y.transpose())
