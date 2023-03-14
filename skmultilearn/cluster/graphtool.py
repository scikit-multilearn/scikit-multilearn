from __future__ import absolute_import
from __future__ import print_function

import graph_tool.all as gt
import numpy as np

from .base import LabelGraphClustererBase
from .helpers import (
    _membership_to_list_of_communities,
    _overlapping_membership_to_list_of_communities,
)


class StochasticBlockModel:
    """A Stochastic Blockmodel fit to Label Graph

    This contains a stochastic block model instance constructed for a block model variant specified in parameters.
    It can be fit to an instance of a graph and set of weights. More information on how to select parameters can be
    found in `the extensive introduction into Stochastic Block Models
    <https://graph-tool.skewed.de/static/doc/demos/inference/inference.html>`_ in graphtool documentation.

    Parameters
    ----------
    nested: boolean
        whether to build a nested Stochastic Block Model or the regular variant,
        will be automatically put under :code:`self.nested`.
    use_degree_correlation: boolean
        whether to correct for degree correlation in modeling, will be automatically
        put under :code:`self.use_degree_correlation`.
    allow_overlap: boolean
        whether to allow overlapping clusters or not, will be automatically
        put under :code:`self.allow_overlap`.
    weight_model: string or None
        decide whether to generate a weighted or unweighted graph,
        will be automatically put under :code:`self.weight_model`.

    Attributes
    ----------
    model_: graph_tool.inference.BlockState or its subclass
        an instance of the fitted model obtained from graph-tool
    """

    def __init__(self, nested, use_degree_correlation, allow_overlap, weight_model):
        self.nested = nested
        self.use_degree_correlation = use_degree_correlation
        self.allow_overlap = allow_overlap
        self.weight_model = weight_model
        self.model_ = None

    def fit_predict(self, graph, weights):
        """Fits model to a given graph and weights list

        Sets :code:`self.model_` to the state of graphtool's Stochastic Block Model the after fitting.

        Attributes
        ----------
        graph: graphtool.Graph
            the graph to fit the model to
        weights: graphtool.EdgePropertyMap<double>
            the property map: edge -> weight (double) to fit the model to, if weighted variant
            is selected

        Returns
        -------
        numpy.ndarray
            partition of labels, each sublist contains label indices
            related to label positions in :code:`y`
        """
        if self.weight_model:
            self.model_ = self._model_fit_function()(
                graph,
                deg_corr=self.use_degree_correlation,
                overlap=self.allow_overlap,
                state_args=dict(recs=[weights], rec_types=[self.weight_model]),
            )
        else:
            self.model_ = self._model_fit_function()(
                graph, deg_corr=self.use_degree_correlation, overlap=self.allow_overlap
            )
        return self._detect_communities()

    def _detect_communities(self):
        if self.nested:
            lowest_level = self.model_.get_levels()[0]
        else:
            lowest_level = self.model_

        number_of_communities = lowest_level.get_B()
        if self.allow_overlap:
            # the overlaps block returns
            # membership vector, and also edges vectors, we need just the membership here at the moment
            membership_vector = list(lowest_level.get_overlap_blocks()[0])
        else:
            membership_vector = list(lowest_level.get_blocks())

        if self.allow_overlap:
            return _overlapping_membership_to_list_of_communities(
                membership_vector, number_of_communities
            )

        return _membership_to_list_of_communities(
            membership_vector, number_of_communities
        )

    def _model_fit_function(self):
        if self.nested:
            return gt.minimize_nested_blockmodel_dl
        else:
            return gt.minimize_blockmodel_dl


class GraphToolLabelGraphClusterer(LabelGraphClustererBase):
    """Fits a Stochastic Block Model to the Label Graph and infers the communities

    This clusterer clusters the label space using by fitting a stochastic block
    model to the label network and inferring the community structure using graph-tool.
    The obtained community structure is returned as the label clustering. More information on the inference itself
    can be found in `the extensive introduction into Stochastic Block Models
    <https://graph-tool.skewed.de/static/doc/demos/inference/inference.html>`_ in graphtool documentation.

    Parameters
    ----------
    graph_builder: a GraphBuilderBase inherited transformer
        the graph builder to provide the adjacency matrix and weight map for the underlying graph
    model: StochasticBlockModel
        the desired stochastic block model variant to use


    Attributes
    ----------
    graph_ : graphtool.Graph
        object representing a label co-occurence graph
    weights_ : graphtool.EdgeProperty<double>
        edge weights defined by graph builder stored in a graphtool compatible format


    .. note ::

        This functionality is still undergoing research.


    .. note ::

        This clusterer is GPL-licenced and will taint your code with GPL restrictions.



    References
    ----------

    If you use this class please cite:

    .. code : latex

        article{peixoto_graph-tool_2014,
         title = {The graph-tool python library},
         url = {http://figshare.com/articles/graph_tool/1164194},
         doi = {10.6084/m9.figshare.1164194},
         urldate = {2014-09-10},
         journal = {figshare},
         author = {Peixoto, Tiago P.},
         year = {2014},
         keywords = {all, complex networks, graph, network, other}}


    Examples
    --------

    An example code for using this clusterer with a classifier looks like this:

    .. code-block:: python

        from sklearn.ensemble import RandomForestClassifier
        from skmultilearn.problem_transform import LabelPowerset
        from skmultilearn.cluster import IGraphLabelGraphClusterer, LabelCooccurrenceGraphBuilder
        from skmultilearn.ensemble import LabelSpacePartitioningClassifier

        # construct base forest classifier
        base_classifier = RandomForestClassifier(n_estimators=1000)

        # construct a graph builder that will include
        # label relations weighted by how many times they
        # co-occurred in the data, without self-edges
        graph_builder = LabelCooccurrenceGraphBuilder(
            weighted = True,
            include_self_edges = False
        )

        # select parameters for the model, we fit a flat,
        # non-degree correlated, partitioning model
        # which will use fit the normal distribution as the weights model
        model = StochasticBlockModel(
            nested=False,
            use_degree_correlation=True,
            allow_overlap=False,
            weight_model='real-normal'
        )

        # setup problem transformation approach with sparse matrices for random forest
        problem_transform_classifier = LabelPowerset(classifier=base_classifier,
            require_dense=[False, False])

        # setup the clusterer to use, we selected the fast greedy modularity-maximization approach
        clusterer = GraphToolLabelGraphClusterer(graph_builder=graph_builder, model=model)

        # setup the ensemble metaclassifier
        classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)

    For more use cases see `the label relations exploration guide <../labelrelations.ipynb>`_.

    """

    def __init__(self, graph_builder, model):
        super(GraphToolLabelGraphClusterer, self).__init__(graph_builder)

        self.model = model
        self.graph_builder = graph_builder

    def fit_predict(self, X, y):
        """Performs clustering on y and returns list of label lists

        Builds a label graph using the provided graph builder's `transform` method
        on `y` and then detects communities using the selected `method`.

        Sets :code:`self.weights_` and :code:`self.graph_`.

        Parameters
        ----------
        X : None
            currently unused, left for scikit compatibility
        y : scipy.sparse
            label space of shape :code:`(n_samples, n_labels)`

        Returns
        -------
        arrray of arrays of label indexes (numpy.ndarray)
            label space division, each sublist represents labels that are in that community
        """
        self._build_graph_instance(y)
        clusters = self.model.fit_predict(self.graph_, weights=self.weights_)
        return np.array([community for community in clusters if len(community) > 0])

    def _build_graph_instance(self, y):
        edge_map = self.graph_builder.transform(y)

        g = gt.Graph(directed=False)
        g.add_vertex(y.shape[1])

        self.weights_ = g.new_edge_property("double")

        for edge, weight in edge_map.items():
            e = g.add_edge(edge[0], edge[1])
            self.weights_[e] = weight

        self.graph_ = g
