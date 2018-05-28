from __future__ import absolute_import

import numpy as np

from .base import LabelSpaceClustererBase
from .helpers import _membership_to_list_of_communities


class MatrixLabelSpaceClusterer(LabelSpaceClustererBase):
    """Cluster the label space using a scikit-compatible matrix-based clusterer"""

    def __init__(self, clusterer=None, pass_input_space=False):
        """Initializes the clusterer

        Attributes
        ----------
        clusterer : sklearn.base.ClusterMixin
            a clonable instance of a scikit-compatible clusterer, will be automatically
            put under :code:`self.clusterer`.
        pass_input_space : bool (default is False)
            whether to take :code:`X` into consideration upon clustering,
            use only if you know that the clusterer can handle two
            parameters for clustering, will be automatically
            put under :code:`self.pass_input_space`.
        """
        super(MatrixLabelSpaceClusterer, self).__init__()

        self.clusterer = clusterer
        self.pass_input_space = pass_input_space

    def fit_predict(self, X, y):
        """Clusters the output space

        The clusterer's :code:`fit_predict` method is executed
        on either X and y.T vectors (if :code:`self.pass_input_space` is true)
        or just y.T to detect clusters of labels.

        The transposition of label space is used to align with
        the format expected by scikit-learn classifiers, i.e. we cluster
        labels with label assignment vectors as samples.


        Returns
        -------
        numpy.ndarray
            partition of labels, each sublist contains label indices
            related to label positions in :code:`y`
        """

        if self.pass_input_space:
            result = self.clusterer.fit_predict(X, y.transpose())
        else:
            result = self.clusterer.fit_predict(y.transpose())
        return np.array(_membership_to_list_of_communities(result, 1 + max(result)))
