from .partition import LabelSpacePartitioningClassifier
import copy
import random
import numpy as np
from scipy import sparse


class RakelD(LabelSpacePartitioningClassifier):
    """Distinct RAndom k-labELsets multi-label classifier."""

    def __init__(self, classifier=None, labelset_size=None, require_dense=None):
        super(RakelD, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self.labelset_size = labelset_size
        self.copyable_attrs = ['labelset_size', 'classifier', 'require_dense']

    def generate_partition(self, X, y):
        """Internal method for sampling k-labELsets"""

        label_sets = []
        self.label_count = y.shape[1]
        free_labels = xrange(self.label_count)
        self.model_count = int(np.ceil(self.label_count / self.labelset_size))

        while len(label_sets) <= self.model_count:
            if len(free_labels) == 0:
                break
            if len(free_labels) < self.labelset_size:
                label_sets.append(free_labels)
                continue

            label_set = random.sample(free_labels, self.labelset_size)
            free_labels = list(set(free_labels).difference(set(label_set)))
            label_sets.append(label_set)

        self.partition = label_sets
