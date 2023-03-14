# -*- coding: utf-8 -*-
from builtins import object

from ..utils import get_matrix_in_format
from sklearn.base import BaseEstimator


class GraphBuilderBase(object):
    """An abstract base class for a graph building class used in Label Space clustering.

    Inherit it in your classifier according to`developer guide <../developer.ipynb>`_.

    """

    def __init__(self):
        super(GraphBuilderBase, self).__init__()

    def transform(self, y):
        """Abstract method for graph edge map builder for a label space clusterer

        Implement it in your classifier according to`developer guide <../developer.ipynb>`_.

        Raises
        ------
        NotImplementedError
            this is an abstract method
        """
        raise NotImplementedError("GraphBuilderBase::transform()")


class LabelSpaceClustererBase(BaseEstimator):
    """An abstract base class for Label Space clustering

    Inherit it in your classifier according to`developer guide <../developer.ipynb>`_.

    """

    def __init__(self):
        super(LabelSpaceClustererBase, self).__init__()

    def fit_predict(self, X, y):
        """Abstract method for clustering label space

        Implement it in your classifier according to`developer guide <../developer.ipynb>`_.

        Raises
        ------
        NotImplementedError
            this is an abstract method
        """
        raise NotImplementedError("LabelSpaceClustererBase::fit_predict()")


class LabelGraphClustererBase(object):
    """An abstract base class for Label Graph clustering

    Inherit it in your classifier according to`developer guide <../developer.ipynb>`_.

    """

    def __init__(self, graph_builder):
        """

        Attributes
        ----------
        graph_builder : a  GraphBuilderBase derivative class
            a graph building class for the clusterer
        """
        super(LabelGraphClustererBase, self).__init__()
        self.graph_builder = graph_builder

    def fit_predict(self, X, y):
        """Abstract method for clustering label space

        Implement it in your classifier according to`developer guide <../developer.ipynb>`_.

        Raises
        ------
        NotImplementedError
            this is an abstract method
        """
        raise NotImplementedError("LabelGraphClustererBase::fit_predict()")


class LabelCooccurrenceGraphBuilder(GraphBuilderBase):
    """Base class providing API and common functions for all label
    co-occurence based multi-label classifiers.

    This graph builder constructs a Label Graph based on the output matrix where two label nodes are connected
    when at least one sample is labeled with both of them. If the graph is weighted, the weight of an edge between two
    label nodes is the number of samples labeled with these two labels. Self-edge weights contain the number of samples
    with a given label.

    Parameters
    ----------
    weighted: bool
        decide whether to generate a weighted or unweighted graph.
    include_self_edges : bool
        decide whether to include self-edge i.e. label 1 - label 1 in
        co-occurrence graph
    normalize_self_edges: bool
        if including self edges, divide the (i, i) edge by 2.0, requires include_self_edges=True


    References
    ----------

    If you use this graph builder please cite the clustering paper:

    .. code:: latex

        @Article{datadriven,
            author = {Szymański, Piotr and Kajdanowicz, Tomasz and Kersting, Kristian},
            title = {How Is a Data-Driven Approach Better than Random Choice in
            Label Space Division for Multi-Label Classification?},
            journal = {Entropy},
            volume = {18},
            year = {2016},
            number = {8},
            article_number = {282},
            url = {http://www.mdpi.com/1099-4300/18/8/282},
            issn = {1099-4300},
            doi = {10.3390/e18080282}
        }


    Examples
    --------

    A full example of building a modularity-based label space division based on the Label Co-occurrence Graph and
    classifying with a separate classifier chain per subspace.

    .. code :: python

        from skmultilearn.cluster import LabelCooccurrenceGraphBuilder, NetworkXLabelGraphClusterer
        from skmultilearn.ensemble import LabelSpacePartitioningClassifier
        from skmultilearn.problem_transform import ClassifierChain
        from sklearn.naive_bayes import GaussianNB

        graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False, normalize_self_edges=False)
        clusterer = NetworkXLabelGraphClusterer(graph_builder, method='louvain')
        classifier = LabelSpacePartitioningClassifier(
            classifier = ClassifierChain(classifier=GaussianNB()),
            clusterer = clusterer
        )
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)


    For more use cases see `the label relations exploration guide <../labelrelations.ipynb>`_.

    """

    def __init__(
        self, weighted=None, include_self_edges=None, normalize_self_edges=None
    ):
        super(LabelCooccurrenceGraphBuilder, self).__init__()

        if weighted not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

        if include_self_edges not in [True, False]:
            raise ValueError(
                "Decision whether to include self edges needs to be a boolean"
            )

        if include_self_edges and (normalize_self_edges not in [True, False]):
            raise ValueError(
                "Decision whether to normalize self edges needs to be a boolean"
            )

        if normalize_self_edges and not include_self_edges:
            raise ValueError(
                "Include self edges must be set to true if normalization is true"
            )

        if normalize_self_edges and not weighted:
            raise ValueError(
                "Normalizing self-edge weights_ does not make sense in an unweighted graph"
            )

        self.is_weighted = weighted
        self.include_self_edges = include_self_edges
        self.normalize_self_edges = normalize_self_edges

    def transform(self, y):
        """Generate adjacency matrix from label matrix

        This function generates a weighted or unweighted co-occurence Label Graph adjacency matrix in dictionary of keys
        format based on input binary label vectors

        Parameters
        ----------
        y : numpy.ndarray or scipy.sparse
            dense or sparse binary matrix with shape
            :code:`(n_samples, n_labels)`

        Returns
        -------
        Dict[(int, int), float]
            weight map with a tuple of label indexes as keys and a the number of samples in which the two co-occurred
        """
        label_data = get_matrix_in_format(y, "lil")
        label_count = label_data.shape[1]
        edge_map = {}

        for row in label_data.rows:
            if self.include_self_edges:
                pairs = [(a, b) for b in row for a in row if a <= b]
            else:
                pairs = [(a, b) for b in row for a in row if a < b]

            for p in pairs:
                if p not in edge_map:
                    edge_map[p] = 1.0
                else:
                    if self.is_weighted:
                        edge_map[p] += 1.0

        if self.normalize_self_edges:
            for i in range(label_count):
                if (i, i) in edge_map:
                    edge_map[(i, i)] = edge_map[(i, i)] / 2.0

        return edge_map
