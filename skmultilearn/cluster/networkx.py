from __future__ import absolute_import

import community
import networkx as nx
import numpy as np
from builtins import range

from .base import LabelSpaceNetworkClustererBase
from .helpers import _membership_to_list_of_communities


class NetworkXLabelCooccurenceClusterer(LabelSpaceNetworkClustererBase):
    """Cluster label space with NetworkX and louvain community detection

    This clusterer builds a NetworkX graph using the graph from a provided
    GraphBuilder instance. It then performs a weighted or unweighted,
    depending on the GraphBuilder

    """

    def __init__(self, graph_builder):
        """Initializes the clusterer

        Attributes
        ----------
        graph_builder: a GraphBuilderBase inherited transformer
                Class used to provide an underlying graph for NetworkX
        """
        super(NetworkXLabelCooccurenceClusterer, self).__init__(graph_builder)

    def fit_predict(self, X, y):
        """Performs clustering on y and returns list of label lists

        Builds a label coocurence_graph using
        :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix`
        on `y` and then detects communities using a selected `method`.

        Parameters
        ----------
        X : currently unused, left for scikit compatibility
        y : scipy.sparse
            label space of shape :code:`(n_samples, n_labels)`

        Returns
        -------
        array of arrays
            numpy array of arrays of label indexes, where each sub-array
            represents labels that are in a separate community
        """
        edge_map = self.graph_builder.transform(y)

        if self.graph_builder.is_weighted:
            self.weights = dict(weight=list(edge_map.values()))
        else:
            self.weights = dict(weight=None)

        self.coocurence_graph = nx.Graph()
        for n in range(y.shape[1]):
            self.coocurence_graph.add_node(n)

        for e, w in edge_map.items():
            self.coocurence_graph.add_edge(e[0], e[1], weight=w)

        self.partition_dict = community.best_partition(self.coocurence_graph)
        self.partition = np.array(
            _membership_to_list_of_communities([self.partition_dict[i] for i in range(y.shape[1])],
                                               1+max(self.partition_dict.values())))
        return self.partition
