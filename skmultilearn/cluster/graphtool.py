from __future__ import absolute_import
from __future__ import print_function

import graph_tool.all as gt
import numpy as np
from builtins import range

from .base import LabelSpaceNetworkClustererBase
from .helpers import _membership_to_list_of_communities, _overlapping_membership_to_list_of_communities

class StochasticBlockModel:
    def __init__(self, nested, use_degree_correlation, allow_overlap, weight_model):
        self.nested = nested
        self.use_degree_correlation = use_degree_correlation
        self.allow_overlap = allow_overlap
        self.weight_model = weight_model
        self._state = None

    def model(self):
        if self.nested:
            return gt.minimize_nested_blockmodel_dl
        else:
            return gt.minimize_blockmodel_dl

    def fit(self, graph, weights):
        if self.weight_model:
            self._state = self.model()(
                graph,
                deg_corr=self.use_degree_correlation,
                overlap=self.allow_overlap,
                state_args=dict(recs=[weights],
                                rec_types=[self.weight_model])
            )
        else:
            self._state = self.model()(
                graph,
                deg_corr=self.use_degree_correlation,
                overlap=self.allow_overlap
            )
        return self

    def entropy(self):
        return self._state.entropy()

    def communities(self):
        if self.nested:
            lowest_level = self._state.get_levels()[0]
        else:
            lowest_level = self._state

        number_of_communities = lowest_level.get_B()
        if self.allow_overlap:
            # the overlaps block returns
            # membership vector, and also edges vectors, we need just the membership here at the moment
            membership_vector = list(lowest_level.get_overlap_blocks()[0])
        else:
            membership_vector = list(lowest_level.get_blocks())

        if self.allow_overlap:
            return _overlapping_membership_to_list_of_communities(membership_vector, number_of_communities)

        return _membership_to_list_of_communities(membership_vector, number_of_communities)


class GraphToolCooccurenceClusterer(LabelSpaceNetworkClustererBase):
    """Clusters the label space using graph tool's stochastic block
    modelling community detection method"""

    def __init__(self, graph_builder, model):
        """Initializes the clusterer

        Attributes
        ----------
        graph_builder: a GraphBuilderBase inherited transformer
                Class used to provide an underlying graph
        model: None or GraphToolModel
                Model to use, if None, provide model_selection_criterium and
                the best model will be used
        model_selection_criterium: 'mean_field' or 'bethe'
                Model selection criterion
        """
        super(GraphToolCooccurenceClusterer, self).__init__(graph_builder)

        self.model = model
        self.graph_builder = graph_builder

    def build_graph_instance(self, y):
        """Constructs the label coocurence graph

        This function constructs a graph-tool :py:class:`graphtool.Graph`
        object representing the label co-occurence graph. Run after 
        :code:`self.edge_map` has been populated using
        :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix`
        on `y` in `fit_predict`.

        The graph is available as self.coocurence_graph, and a weight
        `double` graphtool.PropertyMap on edges is set as self.weights.

        Edge weights are all 1.0 if self.weighted is False, otherwise
        they contain the number of samples that are labelled with the
        two labels present in the edge.

        Returns
        -------
        g : graphtool.Graph 
            object representing a label co-occurence graph
        """

        edge_map = self.graph_builder.transform(y)

        g = gt.Graph(directed=False)
        g.add_vertex(y.shape[1])

        self.weights = g.new_edge_property('double')

        for edge, weight in edge_map.items():
            e = g.add_edge(edge[0], edge[1])
            self.weights[e] = weight

        self.coocurence_graph = g

        return g

    def fit_predict(self, X, y):
        """ Performs clustering on y and returns list of label lists

        Builds a label coocurence_graph using 
        :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix`
        on `y` and then detects communities using graph tool's
        stochastic block modeling.

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
        self.label_count = y.shape[1]
        self.build_graph_instance(y)
        self.model.fit(self.coocurence_graph, weights=self.weights)

        self.label_sets = [community for community in self.model.communities() if len(community) > 0]
        self.model_count = len(self.label_sets)

        return np.array(self.label_sets)
