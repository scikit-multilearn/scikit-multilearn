from __future__ import absolute_import

import numpy as np
from builtins import range
from igraph import Graph

from .base import LabelSpaceNetworkClustererBase


class IGraphLabelCooccurenceClusterer(LabelSpaceNetworkClustererBase):
    """Clusters the label space using igraph community detection methods"""

    METHODS = {
        'fastgreedy': lambda graph, w=None: graph.community_fastgreedy(weights=w).as_clustering(),
        'infomap': lambda graph, w=None: graph.community_infomap(edge_weights=w),
        'label_propagation': lambda graph, w=None: graph.community_label_propagation(weights=w),
        'leading_eigenvector': lambda graph, w=None: graph.community_leading_eigenvector(weights=w),
        'multilevel': lambda graph, w=None: graph.community_multilevel(weights=w),
        'walktrap': lambda graph, w=None: graph.community_walktrap(weights=w).as_clustering(),
    }

    def __init__(self, graph_builder, method):
        """Initializes the clusterer

        Attributes
        ----------
        graph_builder: a GraphBuilderBase inherited transformer
            the graph builder to provide the adjacency matrix and weight map for the underlying graph
        method: boolean
            decide whether to generate a weighted or unweighted graph.
        """
        super(IGraphLabelCooccurenceClusterer, self).__init__(graph_builder)
        self.method = method

        if method not in IGraphLabelCooccurenceClusterer.METHODS:
            raise ValueError(
                "{} not a supported igraph community detection method".format(method))

    def fit_predict(self, X, y):
        """Performs clustering on y and returns list of label lists

        Builds a label coocurence_graph using the provided graph builder's `fit_predict` method
        on `y` and then detects communities using the selected `method`.

        Sets :code:`self.weights` to a dictionary: {'weights': list of weights per label pair edge},
        :code:`self.coocurence_graph` to the igraph Graph object contaning the graph representation
        of graph builders weighted adjacency matrix. Sets :code:`self.partition` to the partition
        obtained using selected community detection method on the constructed Graph.

        Parameters
        ----------
        X : scipy.sparse 
            feature space of shape :code:`(n_samples, n_features)`
        y : scipy.sparse
            label space of shape :code:`(n_samples, n_features)`

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

        self.coocurence_graph = Graph(
            edges=[x for x in edge_map],
            vertex_attrs=dict(name=list(range(1, y.shape[1] + 1))),
            edge_attrs=self.weights
        )

        self.partition = np.array(IGraphLabelCooccurenceClusterer.METHODS[
                                      self.method](self.coocurence_graph, self.weights['weight']))
        return self.partition
