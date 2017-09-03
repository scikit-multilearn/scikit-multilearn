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
        It sets :code:`self.model_count` and :code:`self.label_count`.

        Parameters
        -----------
        X : numpy.ndarray or scipy.sparse
            not used, maintained for API compatibility
        y : numpy.ndarray or scipy.sparse
            binary indicator matrix with label assigments of shape
            :code:`(n_samples, n_labels)`
        """
        self.label_count = y.shape[1]
        self.model_count = len(self.partition)
