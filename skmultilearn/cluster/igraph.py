from __future__ import absolute_import
from .base import LabelCooccurenceClustererBase
import numpy as np
import igraph as ig


class IGraphLabelCooccurenceClusterer(LabelCooccurenceClustererBase):
    """Clusters the label space using igraph community detection methods

    Parameters
    ----------

    method : enum from `IGraphLabelCooccurenceClusterer.METHODS`
        the igraph community detection method that will be used

    weighted: boolean
            Decide whether to generate a weighted or unweighted graph.

    """

    METHODS = {
        'fastgreedy': lambda graph, w = None: np.array(graph.community_fastgreedy(weights=w).as_clustering()),
        'infomap': lambda graph, w = None: np.array(graph.community_infomap(edge_weights=w)),
        'label_propagation': lambda graph, w = None: np.array(graph.community_label_propagation(weights=w)),
        'leading_eigenvector': lambda graph, w = None: np.array(graph.community_leading_eigenvector(weights=w)),
        'multilevel': lambda graph, w = None: np.array(graph.community_multilevel(weights=w)),
        'walktrap': lambda graph, w = None: np.array(graph.community_walktrap(weights=w).as_clustering()),
    }

    def __init__(self, method=None, weighted=None):
        super(IGraphLabelCooccurenceClusterer, self).__init__()
        self.method = method
        self.is_weighted = weighted

        if method not in IGraphLabelCooccurenceClusterer.METHODS:
            raise ValueError(
                "{} not a supported igraph community detection method".format(method))

        if weighted not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

    def fit_predict(self, X, y):
    """Performs clustering on y and returns list of label lists

    Builds a label coocurence_graph using :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix` on `y` and then detects communities using a selected `method`.

    Parameters
    ----------
    X : sparse matrix (n_samples, n_features), feature space, not used in this clusterer
    y : sparse matrix (n_samples, n_labels), label space

    Returns
    -------
    partition: list of lists : list of lists label indexes, each sublist represents labels that are in that community


    """
        self.generate_coocurence_adjacency_matrix(y)

        if self.is_weighted:
            self.weights = dict(weight=self.edge_map.values())
        else:
            self.weights = dict(weight=None)

        self.coocurence_graph = ig.Graph(
            edges=[x for x in self.edge_map],
            vertex_attrs=dict(name=range(1, self.label_count + 1)),
            edge_attrs=self.weights
        )

        partition = IGraphLabelCooccurenceClusterer.METHODS[
            self.method](self.coocurence_graph, self.weights['weight'])
        return partition
