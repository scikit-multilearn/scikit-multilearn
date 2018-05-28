from builtins import object

from ..utils import get_matrix_in_format
from sklearn.base import BaseEstimator

class GraphBuilderBase(object):
    """An abstract base class for a graph building class used in label Space clustering

    Implement it in your classifier according to :doc:`../clusterer`.

    """

    def __init__(self):
        super(GraphBuilderBase, self).__init__()

    def transform(self, y):
        """ Abstract method for graph edge map builder for a label space clusterer

        Implement it in your classifier according to :doc:`../clusterer`.

        Raises
        ------
        NotImplementedError
            this is an abstract method
        """
        raise NotImplementedError("GraphBuilderBase::transform()")


class LabelSpaceClustererBase(BaseEstimator):
    """An abstract base class for Label Space clustering

    Implement it in your classifier according to :doc:`../clusterer`.

    """

    def __init__(self):
        super(LabelSpaceClustererBase, self).__init__()

    def fit_predict(self, X, y):
        """ Abstract method for clustering label space

        Implement it in your classifier according to :doc:`../clusterer`.

        Raises
        ------
        NotImplementedError
            this is an abstract method
        """
        raise NotImplementedError("LabelSpaceClustererBase::fit_predict()")


class LabelSpaceNetworkClustererBase(object):
    """An abstract base class for Label Space clustering

    Implement it in your classifier according to :doc:`../clusterer`.

    """

    def __init__(self, graph_builder):
        """

        Attributes
        ----------
        graph_builder : a  GraphBuilderBase derivative class
            a graph building class for the clusterer
        """
        super(LabelSpaceNetworkClustererBase, self).__init__()
        self.graph_builder = graph_builder

    def fit_predict(self, X, y):
        """ Abstract method for clustering label space

        Implement it in your classifier according to :doc:`../clusterer`.

        Raises
        ------
        NotImplementedError
            this is an abstract method
        """
        raise NotImplementedError("LabelSpaceClustererBase::fit_predict()")


class LabelCooccurenceGraphBuilder(GraphBuilderBase):
    """Base class providing API and common functions for all label
    co-occurence based multi-label classifiers.
    """

    def __init__(self, weighted=None, include_self_edges=None, normalize_self_edges=None):
        """Initializes the clusterer

        Attributes
        ----------
        weighted: bool
            decide whether to generate a weighted or unweighted graph.
        include_self_edges : bool
            decide whether to include self-edge i.e. label 1 - label 1 in
            co-occurrence graph
        normalize_self_edges: bool
            if including self edges, divide the (i, i) edge by 2.0
        """
        super(LabelCooccurenceGraphBuilder, self).__init__()

        if weighted not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

        if include_self_edges not in [True, False]:
            raise ValueError(
                "Decision whether to include self edges needs to be a boolean")

        if normalize_self_edges not in [True, False]:
            raise ValueError("Decision whether to normalize self edges needs to be a boolean")

        if normalize_self_edges and not include_self_edges:
            raise ValueError("Include self edges must be set to true if normalization is true")

        if normalize_self_edges and not weighted:
            raise ValueError("Normalizing self-edge weights does not make sense in an unweighted graph")

        self.is_weighted = weighted
        self.include_self_edges = include_self_edges
        self.normalize_self_edges = normalize_self_edges

    def transform(self, y):
        """Generate adjacency matrix from label matrix

        This function generates a weighted or unweighted co-occurence
        graph based on input binary label vectors
        and sets it to :code:`self.coocurence_graph`

        Parameters
        ----------
        y : numpy.ndarray or scipy.sparse
            dense or sparse binary matrix with shape
            :code:`(n_samples, n_labels)`

        Returns
        -------
        dict
            weight map with a tuple of ints as keys
            and a float value :code:`{ (int, int) : float }`
        """
        label_data = get_matrix_in_format(y, 'lil')
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
