from   ..base import MLClassifierBase
from   ..utils import get_matrix_in_format
import numpy as np
import igraph as ig

class LabelCooccurenceClassifier(MLClassifierBase):
    """Base class providing API and common functions for all label cooccurence based multi-label classifiers.
 
    Parameters
    ----------

    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.

    weighted: boolean
            Decide whether to generate a weighted or unweighted graph.
        
    """
    def __init__(self, classifier = None, weighted = None):
        
        super(LabelCooccurenceClassifier, self).__init__(classifier)
        self.is_weighted = weighted


    def generate_coocurence_graph(self, y):
        """This function generates a weighted or unweighted cooccurence graph based on input binary label vectors 
        and sets it to self.coocurence_graph

        y : array-like of array-likes
            An array-like of binary label vectors.


        Returns
        -------

        self: object
            Return self.
        """
        label_data = get_matrix_in_format(y, 'lil')
        self.label_count = label_data.shape[1]

        edge_map = {}

        for row in label_data.rows:
            pairs = [(a,b) for b in row for a in row if a < b]
            for p in pairs:
                if p not in edge_map:
                    edge_map[p] = 1.0
                else:
                    if self.is_weighted:
                        edge_map[p] += 1.0

        # 
        if self.is_weighted:
            self.weights = edge_map.values()
            self.coocurence_graph = ig.Graph(
                edges        = [x for x in edge_map], 
                vertex_attrs = dict(name   = range(1, self.label_count + 1)), 
                edge_attrs   = dict(weight = self.weights))
        else:
            self.coocurence_graph = ig.Graph(
                edges        = [x for x in edge_map], 
                vertex_attrs = dict(name = range(1, self.label_count + 1)))

        return self