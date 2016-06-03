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
        self.label_count = y.shape[1]
        self.model_count = len(self.partition)
