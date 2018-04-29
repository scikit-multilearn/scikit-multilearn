from __future__ import absolute_import

from builtins import range

import community
import networkx as nx
import numpy as np

from .base import LabelSpaceNetworkClustererBase


class NetworkXLabelCooccurenceClusterer(LabelSpaceNetworkClustererBase):
    """Clusters the label space using igraph community detection methods"""

    def __init__(self, graph_builder):
        """Initializes the clusterer

        Attributes
        ----------
        graph_builder: a GraphBuilderBase inherited transformer
                Class used to provide an underlying graph
        """
        super(NetworkXLabelCooccurenceClusterer, self).__init__(graph_builder)

    def fit_predict(self, X, y):
        """Performs clustering on y and returns list of label lists

        Builds a label coocurence_graph using
        :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix`
        on `y` and then detects communities using a selected `method`.

        Parameters
        ----------
        X : scipy.sparse 
            feature space of shape :code:`(n_samples, n_features)`
        y : scipy.sparse
            label space of shape :code:`(n_samples, n_labels)`

        Returns
        -------
        list of lists
            list of lists label indexes, each sublist represents labels
            that are in that community
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
        self.partition = [self.partition_dict[n] for n in range(y.shape[1])]
        return np.array(self.partition)
