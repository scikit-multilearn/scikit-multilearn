from __future__ import absolute_import
from .base import LabelCooccurenceClustererBase
import numpy as np
import graph_tool.all as gt


class GraphToolCooccurenceClusterer(LabelCooccurenceClustererBase):
    """ Clusters the label space using graph tool's stochastic block modelling community detection method

        Parameters
        ----------

        weighted: boolean
                Decide whether to generate a weighted or unweighted graph.

        allow_overlap: boolean
                Allow overlapping of clusters or not.

    """

    def __init__(self, weighted=None, allow_overlap=False):
        super(LabelCooccurenceClustererBase, self).__init__()
        self.is_weighted = weighted
        self.allow_overlap = allow_overlap

        if weighted not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

        if allow_overlap not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

    def generate_coocurence_graph(self):
        """ Constructs a graph tool Graph object representing the label coocurence_graph

            Run after self.edge_map has been populated using :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix` on `y` in `fit_predict`.

            The graph is available as self.coocurence_graph, and a weight `double` graphtool.PropertyMap on edges is set as self.weights.

            Edge weights are all 1.0 if self.weighted is false, otherwise they contain the number of samples that are labelled with the two labels present in the edge.

            Returns
            -------

            g : graphtool.Graph object representing a label co-occurence graph
        """
        g = gt.Graph(directed=False)
        g.add_vertex(self.label_count)

        self.weights = g.new_edge_property('double')

        for edge, weight in self.edge_map.iteritems():
            e = g.add_edge(edge[0], edge[1])
            if self.is_weighted:
                self.weights[e] = weight
            else:
                self.weights[e] = 1.0

        self.coocurence_graph = g

        return g

    def fit_predict(self, X, y):
        """ Performs clustering on y and returns list of label lists

            Builds a label coocurence_graph using :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix` on `y` and then detects communities using graph tool's stochastic block modeling.

            Parameters
            ----------
            X : sparse matrix (n_samples, n_features), feature space, not used in this clusterer
            y : sparse matrix (n_samples, n_labels), label space

            Returns
            -------
            partition: list of lists : list of lists label indexes, each sublist represents labels that are in that community
        """
        self.generate_coocurence_adjacency_matrix(y)
        self.generate_coocurence_graph()

        d = gt.minimize_blockmodel_dl(
            self.coocurence_graph, overlap=self.allow_overlap, ec=self.weights)
        A = d.get_blocks().a

        self.label_sets = [[] for i in xrange(d.B)]
        for k in xrange(len(A)):
            self.label_sets[A[k]].append(k)

        self.model_count = len(self.label_sets)

        return np.array(self.label_sets)
