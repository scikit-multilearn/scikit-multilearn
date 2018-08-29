from __future__ import absolute_import

import numpy as np

from .base import LabelSpaceClustererBase
from .helpers import _membership_to_list_of_communities


class MatrixLabelSpaceClusterer(LabelSpaceClustererBase):
    """Cluster the label space using a scikit-compatible matrix-based clusterer

    Parameters
    ----------
    clusterer : sklearn.base.ClusterMixin
        a clonable instance of a scikit-compatible clusterer, will be automatically
        put under :code:`self.clusterer`.
    pass_input_space : bool (default is False)
        whether to take :code:`X` into consideration upon clustering,
        use only if you know that the clusterer can handle two
        parameters for clustering, will be automatically
        put under :code:`self.pass_input_space`.


    Example code for using this clusterer looks like this:

    .. code-block:: python

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
        from skmultilearn.problem_transform import LabelPowerset
        from skmultilearn.cluster import MatrixLabelSpaceClusterer
        from skmultilearn.ensemble import LabelSpacePartitioningClassifier

        # construct base forest classifier
        base_classifier = RandomForestClassifier(n_estimators=1030)

        # setup problem transformation approach with sparse matrices for random forest
        problem_transform_classifier = LabelPowerset(classifier=base_classifier,
            require_dense=[False, False])

        # setup the clusterer
        clusterer = MatrixLabelSpaceClusterer(clusterer=KMeans(n_clusters=3))

        # setup the ensemble metaclassifier
        classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)
    """

    def __init__(self, clusterer=None, pass_input_space=False):
        super(MatrixLabelSpaceClusterer, self).__init__()

        self.clusterer = clusterer
        self.pass_input_space = pass_input_space

    def fit_predict(self, X, y):
        """Clusters the output space

        The clusterer's :code:`fit_predict` method is executed
        on either X and y.T vectors (if :code:`self.pass_input_space` is true)
        or just y.T to detect clusters of labels.

        The transposition of label space is used to align with
        the format expected by scikit-learn classifiers, i.e. we cluster
        labels with label assignment vectors as samples.

        Returns
        -------
        arrray of arrays of label indexes (numpy.ndarray)
            label space division, each sublist represents labels that are in that community
        """

        if self.pass_input_space:
            result = self.clusterer.fit_predict(X, y.transpose())
        else:
            result = self.clusterer.fit_predict(y.transpose())
        return np.array(_membership_to_list_of_communities(result, 1 + max(result)))
