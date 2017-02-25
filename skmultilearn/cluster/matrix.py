from __future__ import absolute_import
from builtins import range
from .base import LabelSpaceClustererBase
import numpy as np


class MatrixLabelSpaceClusterer(LabelSpaceClustererBase):

    """Clusters the label space using a matrix-based clusterer

        :param clusterer: a clonable instance of a
            `scikit-compatible matrix-based <http://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html>`_ clusterer

        :param pass_input_space bool: whether to take ``X`` into
            consideration upon clustering, use only if you know that the clusterer
            can handle two parameters for clustering

    """

    def __init__(self, clusterer=None, pass_input_space=False):
        super(MatrixLabelSpaceClusterer, self).__init__()

        self.clusterer = clusterer
        self.pass_input_space = pass_input_space

    def fit_predict(self, X, y):
        """ Cluster the output space

        Uses the ``fit_predict`` method of provided ``clusterer``
        to perform label space division.

        :returns: partition of labels, each sublist contains
            label indices related to label positions in ``y``

        :rtype: nd.array of nd.arrays

        :returns: this is just an abstract method

        """

        if self.pass_input_space:
            return self.clusterer.fit_predict(X, y.transpose())

        return self.clusterer.fit_predict(y.transpose())
