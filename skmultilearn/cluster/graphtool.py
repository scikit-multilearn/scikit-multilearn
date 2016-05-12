from __future__ import absolute_import
from .base import LabelCooccurenceClustererBase
import numpy as np
import graph_tool.all as gt


class GraphToolCooccurenceClusterer(LabelCooccurenceClustererBase):
    """Base class providing API and common functions for all label cooccurence based multi-label classifiers.
 
    Parameters
    ----------

    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.

    weighted: boolean
            Decide whether to generate a weighted or unweighted graph.

    allow_overlap: boolean
            Allow overlapping of clusters or not.
        
    """


    def __init__(self, weighted = None, allow_overlap = False):            
        super(IGraphLabelCooccurenceClusterer, self).__init__()
        self.method = method
        self.is_weighted = weighted
        self.allow_overlap = allow_overlap

        if weighted not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

        if allow_overlap not in [True, False]:
            raise ValueError("Weighted needs to be a boolean")

    def generate_coocurence_graph(self):
        g = gt.Graph(directed = False)
        g.add_vertex(self.label_count)
        
        self.weights = g.new_edge_property('double')

        for edge, weight in edge_map.iteritems():
            e = g.add_edge(edge[0], edge[1])
            if self.is_weighted:
                self.weights[e] = weight
            else:
                self.weights[e] = 1.0

        self.coocurence_graph = g


    def fit_predict(self, X, y):
        self.generate_coocurence_adjacency_matrix(y)
        self.generate_coocurence_graph()

        d = gt.minimize_blockmodel_dl(g, overlap = self.allow_overlap, ec = self.weights)
        A = d.get_blocks().a

        self.label_sets = [[] for i in xrange(d.B)]
        for k in xrange(len(A)):
            self.label_sets[A[k]].append(k)
    
        self.model_count = len(self.label_sets)

        return np.array(self.label_sets)
