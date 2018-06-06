import numpy as np
from builtins import range
from builtins import zip
from scipy import sparse

from .partition import LabelSpacePartitioningClassifier


class MajorityVotingClassifier(LabelSpacePartitioningClassifier):
    """Majority Voting ensemble classifier

    Divides the label space using provided clusterer class, trains a provided base classifier
    type classifier for each subset and assign a label to an instance
    if more than half of all classifiers (majority) from clusters that contain the label
    assigned the label to the instance.
    """

    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        """

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
        clusterer : skmultilearn.cluster.base
            the clusterer that divides the label space into clusters, will be automatically
            put under :code:`self.clusterer`
        """
        super(MajorityVotingClassifier, self).__init__(
            classifier=classifier, clusterer=clusterer, require_dense=require_dense
        )

    def predict(self, X):
        """Predict label assignments for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of float
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """
        predictions = [
            self.ensure_input_format(self.ensure_input_format(
                c.predict(X)), sparse_format='csc', enforce_sparse=True)
            for c in self.classifiers
        ]

        voters = np.zeros(self.label_count, dtype='int')
        votes = sparse.lil_matrix(
            (predictions[0].shape[0], self.label_count), dtype='int')
        for model in range(self.model_count):
            for label in range(len(self.partition[model])):
                votes[:, self.partition[model][label]] = votes[
                                                         :, self.partition[model][label]] + predictions[model][:, label]
                voters[self.partition[model][label]] += 1

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = np.round(
                votes[row, column] / float(voters[column]))

        return self.ensure_output_format(votes, enforce_sparse=False)

    def predict_proba(self, X):
        raise NotImplemented("The voting scheme does not define a method for calculating probabilities")