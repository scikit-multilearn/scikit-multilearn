from .partition import LabelSpacePartitioningClassifier


class FixedLabelPartitionClassifier(LabelSpacePartitioningClassifier):
    """Classify for each cluster separately given a fixed label space partition"""

    def __init__(self, classifier=None, require_dense=None, partition=None):
        """Initialize the classifier

        Attributes
        ----------
        classifier : sklearn.base
            the base classifier that will be used in a class, will be
            automatically put under :code:`self.classifier` for future
            access.
        require_dense : [bool, bool]
            whether the base classifier requires [input, output] matrices
            in dense representation, will be automatically
            put under :code:`self.require_dense`
        partition : array of arrays of int from range(0, label_count)
            provided partition of the label space in the for of numpy array of
            numpy arrays of indexes for each partition, will be
            automatically put under :code:`self.partition`
        """
        super(FixedLabelPartitionClassifier, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self.partition = partition
        self.copyable_attrs = ['partition', 'classifier', 'require_dense']

    def generate_partition(self, X, y):
        """Apply the partition to the label space

        Mock function, the partition is assigned in the constructor.
        It sets :code:`self.model_count` to partition size
        and :code:`self.label_count` to number of labels.

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
        return self.partition
