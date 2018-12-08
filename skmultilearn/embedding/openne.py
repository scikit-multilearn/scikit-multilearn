from copy import copy
from openne.gf import GraphFactorization
from openne.graph import Graph
from openne.grarep import GraRep
from openne.hope import HOPE
from openne.lap import LaplacianEigenmaps
from openne.line import LINE
from openne.node2vec import Node2vec
from openne.tadw import TADW
import networkx as nx
import numpy as np


class OpenNetworkEmbedder:
    """Embed the label space using a label network embedder from OpenNE

    Parameters
    ----------
    graph_builder: a GraphBuilderBase inherited transformer
        the graph builder to provide the adjacency matrix and weight map for the underlying graph
    embedding : string
        which of the OpenNE

        +----------------------+--------------------------------------------------------------------------------+
        | Method name string   |                             Description                                        |
        +----------------------+--------------------------------------------------------------------------------+
        | GraRep_               | Detecting communities with largest modularity using incremental greedy search  |
        +----------------------+--------------------------------------------------------------------------------+
        | HOPE_                 | Detecting communities from multiple async label propagation on the graph       |
        +----------------------+--------------------------------------------------------------------------------+
        | LaplacianEigenmaps_   | Detecting communities from multiple async label propagation on the graph       |
        +----------------------+--------------------------------------------------------------------------------+
        | LINE_   | LINE: Large-scale information network embedding       |
        +----------------------+--------------------------------------------------------------------------------+
        | LLE_ | Detecting communities from multiple async label propagation on the graph       |
        +----------------------+--------------------------------------------------------------------------------+
        | node2vec_   | Detecting communities from multiple async label propagation on the graph       |
        +----------------------+--------------------------------------------------------------------------------+
        | TADW_   | Detecting communities from multiple async label propagation on the graph       |
        +----------------------+--------------------------------------------------------------------------------+

        .. _GraRep: https://github.com/thunlp/OpenNE/blob/master/src/openne/grarep.py
        .. _HOPE: https://github.com/thunlp/OpenNE/blob/master/src/openne/hope.py
        .. _LaplacianEigenmaps: https://github.com/thunlp/OpenNE/blob/master/src/openne/lap.py
        .. _LINE: https://github.com/thunlp/OpenNE/blob/master/src/openne/line.py
        .. _LLE: https://github.com/thunlp/OpenNE/blob/master/src/openne/lle.py
        .. _node2vec: https://github.com/thunlp/OpenNE/blob/master/src/openne/node2vec.py
        .. _TADW: https://github.com/thunlp/OpenNE/blob/master/src/openne/tadw.py


    dimension: int
        the dimension of the label embedding vectors
    aggregation_function: 'add', 'multiply', 'average' or Callable
        the function used to aggregate label vectors for all labels assigned to each of the samples
    normalize_weights: boolean
        whether to normalize weights in the label graph by the number of samples or not
    param_dict
        parameters passed to the embedder, don't use the dimension and graph parameters, this class will set them at fit

    Example code for using this embedder looks like this:

    .. code-block:: python

        from skmultilearn.embedding import OpenNetworkEmbedder, EmbeddingClassifier
        from sklearn.ensemble import RandomForestRegressor
        from skmultilearn.adapt import MLkNN
        from skmultilearn.cluster import LabelCooccurrenceGraphBuilder

        graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
        openne_line_params = dict(batch_size=1000, negative_ratio=5)

        clf = EmbeddingClassifier(
            OpenNetworkEmbedder(graph_builder, 'LINE', 4, 'add', True, openne_line_params),
            RandomForestRegressor(n_estimators=10),
            MLkNN(k=5)
        )

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
    """

    _EMBEDDINGS = {
        'GraphFactorization': (GraphFactorization, 'dim'),
        'GraRep': (GraRep, 'dim'),
        'HOPE': (HOPE, 'd'),
        'LaplacianEigenmaps': (LaplacianEigenmaps, 'rep_size'),
        'LINE': (LINE, 'rep_size'),
        'node2vec': (Node2vec, 'dim'),
        'TADW': (TADW, 'dim')
    }

    _AGGREGATION_FUNCTIONS = {
        'add': np.add.reduce,
        'multiply': np.multiply.reduce,
        'average': lambda x: np.average(x, axis=0),
    }

    def __init__(self, graph_builder, embedding, dimension, aggregation_function, normalize_weights, param_dict):
        if embedding not in self._EMBEDDINGS:
            raise ValueError('Embedding must be one of {}'.format(', '.join(self._EMBEDDINGS.keys())))

        if aggregation_function in self._AGGREGATION_FUNCTIONS:
            self.aggregation_function = self._AGGREGATION_FUNCTIONS[aggregation_function]
        elif callable(aggregation_function):
            self.aggregation_function = aggregation_function
        else:
            raise ValueError('Aggregation function must be callable or one of {}'.format(
                ', '.join(self._AGGREGATION_FUNCTIONS.keys()))
            )

        self.embedding = self._EMBEDDINGS[embedding]
        self.param_dict = param_dict
        self.dimension = dimension
        self.graph_builder = graph_builder
        self.normalize_weights = normalize_weights

    def fit(self, X, y):
        """Fits the embedder to data

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        self
            fitted instance of self
        """
        self.fit_transform(X, y)

    def fit_transform(self, X, y):
        """Fit the embedder and transform the output space

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        X, y_embedded
            results of the embedding, input and output space
        """

        self._init_openne_graph(y)
        embedding_class, dimension_key = self._EMBEDDINGS['LINE']
        param_dict = copy(self.param_dict)
        param_dict['graph'] = self.graph_
        param_dict[dimension_key] = self.dimension
        self.embeddings_ = embedding_class(**param_dict)
        return self.embeddings_

    def _init_openne_graph(self, y):
        self.graph_ = Graph()
        self.graph_.G = nx.DiGraph()
        for (src, dst), w in self.graph_builder.transform(y).items():
            self.graph_.G.add_edge(src, dst)
            self.graph_.G.add_edge(dst, src)
            if self.normalize_weights:
                w = float(w) / y.shape[0]
            self.graph_.G[src][dst]['weight'] = w
            self.graph_.G[dst][src]['weight'] = w
        self.graph_.encode_node()
