from __future__ import absolute_import

import numpy as np

from sklearn.base import BaseEstimator

class SKLearnEmbedder(BaseEstimator):
    """Cluster the label space using a scikit-compatible matrix-based clusterer

    Parameters
    ----------
    embedder : sklearn.base.BaseEstimator
        a clonable instance of a scikit-compatible embedder, will be automatically
        put under :code:`self.embedder`, see .
    pass_input_space : bool (default is False)
        whether to take :code:`X` into consideration upon clustering,
        use only if you know that the embedder can handle two
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

    def __init__(self, embedder=None, pass_input_space=False):
        super(BaseEstimator, self).__init__()

        self.embedder = embedder
        self.pass_input_space = pass_input_space

    def fit(self, X,y):
        self.embedder.fit(X, y)

    def fit_transform(self, X, y):
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
            result = self.embedder.fit_transform(X, y)
        else:
            result = self.embedder.fit_transform(y)

        return X, result
