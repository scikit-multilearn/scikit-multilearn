from builtins import object
from ..utils import get_matrix_in_format


class LabelSpaceClustererBase(object):

    """An abstract base class for Label Space clustering

    Implement it in your classifier according to :doc:`../clusterer`.

    """

    def __init__(self):
        super(LabelSpaceClustererBase, self).__init__()

    def fit_predict(self, X, y):
        """ Abstract method for clustering label space

        Implement it in your classifier according to :doc:`../clusterer`.

        :raises NotImplementedError: this is just an abstract method

        """
        raise NotImplementedError("LabelSpaceClustererBase::fit_predict()")


class LabelCooccurenceClustererBase(LabelSpaceClustererBase):

    """Base class providing API and common functions for all label cooccurence based multi-label classifiers.

    Parameters
    ----------

    weighted: boolean
            Decide whether to generate a weighted or unweighted graph.

    include_self_edges : boolean
            Decide whether to include self-edge i.e. label 1 - label 1 in co-occurrence graph

    """

    def __init__(self, weighted=None, include_self_edges=None):
        super(LabelCooccurenceClustererBase, self).__init__()

        self.is_weighted = weighted
        self.include_self_edges = include_self_edges

        if weighted not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

        if include_self_edges not in [True, False]:
            raise ValueError(
                "Decision about whether to include self edges needs to be a boolean")

    def generate_coocurence_adjacency_matrix(self, y):
        """Generate adjacency matrix from label matrix

        This function generates a weighted or unweighted cooccurence graph based on input binary label vectors
        and sets it to self.coocurence_graph

        :param y: binary indicator matrix with label assignments
        :type y: dense or sparse matrix of {0, 1} (n_samples, n_labels)


        Returns
        -------

        edge_map: dict{ (int, int) : float }
            Returns a dict of weights

        """
        label_data = get_matrix_in_format(y, 'lil')
        self.label_count = label_data.shape[1]
        edge_map = {}

        for row in label_data.rows:
            pairs = None
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

        for i in range(self.label_count):
            if (i,i) in edge_map:
                edge_map[(i,i)] = edge_map[(i,i)]/2.0

        self.edge_map = edge_map
        return edge_map
