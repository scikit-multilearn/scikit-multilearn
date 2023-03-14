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

    Parameters
    ----------
    classifier : :class:`~sklearn.base.BaseEstimator`
        the base classifier that will be used in a class, will be
        automatically put under :code:`self.classifier`.
    clusterer : :class:`~skmultilearn.cluster.LabelSpaceClustererBase`
        object that partitions the output space, will be
        automatically put under :code:`self.clusterer`.
    require_dense : [bool, bool]
        whether the base classifier requires [input, output] matrices
        in dense representation, will be automatically
        put under :code:`self.require_dense`.


    Attributes
    ----------
    model_count_ : int
        number of trained models, in this classifier equal to the number of partitions
    partition_ : List[List[int]], shape=(`model_count_`,)
        list of lists of label indexes, used to index the output space matrix, set in :meth:`_generate_partition`
        via :meth:`fit`
    classifiers : List[:class:`~sklearn.base.BaseEstimator`], shape=(`model_count_`,)
        list of classifiers trained per partition, set in :meth:`fit`


    Examples
    --------
    Here's an example of building an overlapping ensemble of chains

    .. code :: python

        from skmultilearn.ensemble import MajorityVotingClassifier
        from skmultilearn.cluster import FixedLabelSpaceClusterer
        from skmultilearn.problem_transform import ClassifierChain
        from sklearn.naive_bayes import GaussianNB


        classifier = MajorityVotingClassifier(
            clusterer = FixedLabelSpaceClusterer(clusters = [[1,2,3], [0, 2, 5], [4, 5]]),
            classifier = ClassifierChain(classifier=GaussianNB())
        )
        classifier.fit(X_train,y_train)
        predictions = classifier.predict(X_test)

    More advanced examples can be found in `the label relations exploration guide <../labelrelations.ipynb>`_

    """

    def __init__(self, classifier=None, clusterer=None, require_dense=None):
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
            self._ensure_input_format(
                self._ensure_input_format(c.predict(X)),
                sparse_format="csc",
                enforce_sparse=True,
            )
            for c in self.classifiers_
        ]

        voters = np.zeros(self._label_count, dtype="int")
        votes = sparse.lil_matrix(
            (predictions[0].shape[0], self._label_count), dtype="int"
        )
        for model in range(self.model_count_):
            for label in range(len(self.partition_[model])):
                votes[:, self.partition_[model][label]] = (
                    votes[:, self.partition_[model][label]]
                    + predictions[model][:, label]
                )
                voters[self.partition_[model][label]] += 1

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = np.round(votes[row, column] / float(voters[column]))

        return self._ensure_output_format(votes, enforce_sparse=False)

    def predict_proba(self, X):
        raise NotImplemented(
            "The voting scheme does not define a method for calculating probabilities"
        )
