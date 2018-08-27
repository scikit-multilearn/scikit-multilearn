from __future__ import absolute_import
from .base import LabelSpaceClustererBase

class FixedLabelSpaceClusterer(LabelSpaceClustererBase):
    """Return a fixed label space partition

    This clusterer takes a predefined fixed ``clustering`` of the label space and returns it in fit_predict as the label
    space division. This is useful for employing expert knowledge about label space division or partitions in ensemble
    classifiers such as: :class:`LabelSpacePartitioningClassifier` or :class:`MajorityVotingClassifier`.

    Attributes
    ----------
    clusters: array of arrays of int from range(0, label_count)
        provided partition of the label space in the for of numpy array of
        numpy arrays of indexes for each partition, will be
        automatically put under :code:`self.cluster`

    An example use of the fixed clusterer with a label partitioning classifier to train randomforests for a set of
    subproblems defined upon expert knowledge:

    .. code :: python

        from skmultilearn.ensemble import LabelSpacePartitioningClassifier
        from skmultilearn.cluster import FixedLabelSpaceClusterer
        from skmultilearn.problem_transform import LabelPowerset
        from sklearn.ensemble import RandomForestClassifier

        classifier = LabelSpacePartitioningClassifier(
            classifier = LabelPowerset(
                classifier=RandomForestClassifier(n_estimators=100),
                require_dense = [False, True]
            ),
            require_dense = [True, True],
            clusterer = FixedLabelSpaceClusterer(clustering=[[1,2,3], [0,4]])
        )

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)

    """

    def __init__(self, clusters=None):
        super(FixedLabelSpaceClusterer, self).__init__()

        self.clusters = clusters

    def fit_predict(self, X, y):
        """Returns the provided label space division

        Returns
        -------
        numpy.ndarray
            partition of labels, each sublist contains label indices
            related to label positions in :code:`y`
        """

        return self.clusters
