from .partition import LabelSpacePartitioningClassifier
import copy
import random
import numpy as np
from scipy import sparse


class FixedLabelPartitionClassifier(LabelSpacePartitioningClassifier):
    """Classify given a fixed Label Space partition"""

    def __init__(self, classifier=None, require_dense=None, partition=None):
        super(FixedLabelPartitionClassifier, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self.partition = partition
        self.copyable_attrs = ['partition', 'classifier', 'require_dense']

    def generate_partition(self, X, y):
    	"""Assign fixed partition of the label space

    	Mock function, the partition is assigned in the constructor.

        :param X: not used, maintained for api compatibility
        :param y: binary indicator matrix with label assignments
        :type y: dense or sparse matrix of {0, 1} (n_samples, n_labels)

        Sets `self.model_count` and `self.label_count`.

        """
        self.label_count = y.shape[1]
        self.model_count = len(self.partition)
