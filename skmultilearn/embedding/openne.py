from copy import copy
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
    EMBEDDINGS = {
        'GraRep': (GraRep, 'dim'),
        'HOPE': (HOPE, 'd'),
        'LaplacianEigenmaps': (LaplacianEigenmaps, 'rep_size'),
        'LINE': (LINE, 'rep_size'),
        'Node2vec': (Node2vec, 'dim'),
        'TADW': (TADW, 'dim')
    }

    AGGREGATION_FUNCTIONS = {
        'add': np.add.reduce,
        'mul': np.multiply.reduce,
        'avg': lambda x: np.average(x, axis=0),
    }

    def __init__(self, graph_builder, embedding, dimension, aggregation_function, normalize_weights, param_dict):
        if embedding not in self.EMBEDDINGS:
            raise ValueError('Embedding must be one of {}'.format(', '.join(self.EMBEDDINGS.keys())))

        if aggregation_function in self.AGGREGATION_FUNCTIONS:
            self.aggregation_function = self.AGGREGATION_FUNCTIONS[aggregation_function]
        elif callable(aggregation_function):
            self.aggregation_function = aggregation_function
        else:
            raise ValueError('Aggregation function must be callable or one of {}'.format(
                ', '.join(self.AGGREGATION_FUNCTIONS.keys()))
            )

        self.embedding = self.EMBEDDINGS[embedding]
        self.param_dict = param_dict
        self.dimension = dimension
        self.graph_builder = graph_builder
        self.normalize_weights = normalize_weights

    def fit(self, X, y):
        self.fit_transform(X, y)

    def fit_transform(self, X, y):
        self._init_openne_graph(y)
        embedding_class, dimension_key = self.EMBEDDINGS['LINE']
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
