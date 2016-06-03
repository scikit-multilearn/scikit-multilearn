.. _implementclusterer:
Developing a label space clusterer
==================================

One of the approaches to multi-label classification is to cluster the label space into subspaces and perform classification in smaller subproblems to reduce the risk of under/overfitting.


``scikit-multilearn`` follows `scikit-learn's ClustererMixin interface <http://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html#sklearn.base.ClusterMixin>`_ API for a clustering class. ``scikit-learn`` concentrates on single-label classification in which one usually performs clustering of the input space ``X`` and not the output class space. In ``scikit-learn`` the clusterer class is expected to take ``X`` and ``y`` and provide a clustering of ``X`` as an ``ndarray`` of ``n_samples`` elements, each element corresponding to the id of the cluster to which the observation is assigned, thus if sample no. 3 is assigned to cluster no. 1: ``result[3]`` should be equal to 1.

The clusterer for the label space in `scikit-multilearn` follows this interface, in order to create your own label space clusterer you need to inherit :class:`LabelSpaceClustererBase` and implement the ``fit_predict(X, y)`` class method. Expect ``X`` and ``y`` to be sparse matrices, you and also use :func:`skmultilearn.utils.get_matrix_in_format` to convert to a desired matrix format. ``fit_predict(X, y)`` should return an array-like, preferably an ``ndarray`` or ``list`` of ``n_labels`` of integers indicating the no. of cluster a given label is assigned to similarly as it is performed in ``scikit-learn`` clusterers.


Example classifier
------------------

Let us look at a toy example, where a clusterer divides the label space based on how a given label's ordinal divides modulo a given number of clusters.

.. code-block:: python

    from skmultilearn.ensemble import LabelCooccurenceClustererBase
        
    class ModuloClusterer(LabelSpaceClustererBase):
        def __init__(self, number_of_clusters = None):
            super(ModuloClusterer, self).__init__()
            self.number_of_clusters = number_of_clusters

        def fit_predict(self, X, y):
            number_of_labels = y.shape[1]
            # assign a label to a cluster no. label ordinal %  number of labeSls 
            return map(lambda x: x % self.number_of_clusters, xrange(number_of_labels))


Label co-occurence graph
------------------------

A feature present currently only in ``scikit-multilearn`` is the possibility to divide the label space based on analysing a label co-occurence graph. In such a graph labels are represented as nodes, and edges are generated based on how labels co-occur together among samples. In other words an edge between label no. ``a`` and number ``b`` is present if there exists a sample in ``X`` that is labeled with both ``a`` and ``b``. The :class:`LabelCooccurenceClustererBase` provides a method :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix(X, y)` which generates a dict containing label pairs as keys, and a float as edge weight value:

- ``1.0`` in an unweighted setting
- number of samples in ``X`` that are labeled with both labels in a weighted setting (when ``self.is_weighted`` is ``True``)

The `edge_map` is both returned and stored in the class as ``self.edge_map``.  This is a base class for building label co-occurence graphs. Subclass ``LabelCooccurenceClustererBase`` and use  ``generate_coocurence_adjacency_matrix`` at the beginning of your ``fit_predict`` as shown below, than build a graph using the `edge_map` property, and infer the communities from the graph. Interfaces for two popular Python graph libraries already exist: 

- :class:`skmultilearn.ensemble.GraphToolCooccurenceClusterer` that constructs the graph-tool graph object and uses stochastic block modelling for clustering
- :class:`skmultilearn.ensemble.IGraphLabelCooccurenceClusterer` that constructs an igraph graph object and allows the use of a variety of igraph's community detection methods for clustering

to use them, just subclass the class and start your fit_predict method like this:

.. code-block:: python

        def fit_predict(self, X, y):
            self.generate_coocurence_adjacency_matrix(y)
            self.generate_coocurence_graph()

Your graph object (a `graphtool Graph <https://graph-tool.skewed.de/static/doc/quickstart.html>`_ or an `igraph Graph <http://igraph.org/python/doc/igraph.Graph-class.html>`_) is available at ``self.coocurence_graph`` after those two lines.