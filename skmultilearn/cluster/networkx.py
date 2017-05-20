from __future__ import absolute_import
from builtins import range
from .base import LabelCooccurenceClustererBase
import numpy as np
import community
import networkx as nx

class NetworkXLabelCooccurenceClusterer(LabelCooccurenceClustererBase):

    """Clusters the label space using igraph community detection methods

    Parameters
    ----------

    method : enum from `IGraphLabelCooccurenceClusterer.METHODS`
        the igraph community detection method that will be used

    weighted: boolean
            Decide whether to generate a weighted or unweighted graph.

    include_self_edges : boolean
            Decide whether to include self-edge i.e. label 1 - label 1 in co-occurrence graph

    """

    def __init__(self, weighted=None, include_self_edges=None):
        super(NetworkXLabelCooccurenceClusterer, self).__init__(
            weighted=weighted, include_self_edges=include_self_edges)


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
            self.weights = dict(weight=list(self.edge_map.values()))
        else:
            self.weights = dict(weight=None)

        self.coocurence_graph=nx.Graph()
        for n in range(y.shape[1]):
            self.coocurence_graph.add_node(n)

        for e, w in self.edge_map.iteritems():
            self.coocurence_graph.add_edge(e[0],e[1],weight=w)

        self.partition_dict = community.best_partition(self.coocurence_graph)
        self.partition = [self.partition_dict[n] for n in range(y.shape[1])]
        return np.array(self.partition)
