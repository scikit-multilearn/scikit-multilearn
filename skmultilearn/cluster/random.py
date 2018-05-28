from __future__ import absolute_import

import numpy as np
import random

from .base import LabelSpaceClustererBase
from .helpers import _membership_to_list_of_communities


class RandomLabelSpaceClusterer(LabelSpaceClustererBase):
    """Clusters the label space using a matrix-based clusterer"""

    def __init__(self, partition_size, partition_count, allow_overlap):
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
        super(RandomLabelSpaceClusterer, self).__init__()

        self.partition_size = partition_size
        self.partition_count = partition_count
        self.allow_overlap = allow_overlap

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

        label_sets = []
        self.label_count = y.shape[1]
        free_labels = range(self.label_count)

        while len(label_sets) <= self.partition_count:
            if not self.allow_overlap:
                if len(free_labels) == 0:
                    break

                # in this case, we are unable to draw new labels, add all that remain
                if len(free_labels) < self.partition_size:
                    label_sets.append(free_labels)
                    free_labels = list(set(free_labels).difference(set(label_set)))
                    assert len(free_labels) == 0
                    break

            label_set = random.sample(free_labels, self.partition_size)
            if not self.allow_overlap:
                free_labels = list(set(free_labels).difference(set(label_set)))

            if label_set not in label_sets:
                label_sets.append(label_set)

        self.partition = label_sets
        return np.array(self.partition)