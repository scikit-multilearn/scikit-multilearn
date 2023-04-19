from copy import copy
from openne.gf import GraphFactorization
from openne.graph import Graph
from openne.grarep import GraRep
from openne.hope import HOPE
from openne.lap import LaplacianEigenmaps
from openne.line import LINE
from openne.lle import LLE
import networkx as nx
import numpy as np
import tensorflow as tf
import scipy.sparse as sp


class OpenNetworkEmbedder:
    """Embed the label space using a label network embedder from OpenNE

    Implements an OpenNE based LNEMLC: label network embeddings for multi-label classification.

    Parameters
    ----------
    graph_builder: a GraphBuilderBase inherited transformer
        the graph builder to provide the adjacency matrix and weight map for the underlying graph
    embedding : string, one of {'GraphFactorization', 'GraRep', 'HOPE', 'LaplacianEigenmaps', 'LINE', 'LLE'}
        the selected OpenNE_ embedding

        +----------------------+--------------------------------------------------------------------------------+
        | Method name string   |                             Description                                        |
        +----------------------+--------------------------------------------------------------------------------+
        | GraphFactorization_  | Graph factorization embeddings                                                 |
        +----------------------+--------------------------------------------------------------------------------+
        | GraRep_              | Graph representations with global structural information                       |
        +----------------------+--------------------------------------------------------------------------------+
        | HOPE_                | High-order Proximity Preserved Embedding                                       |
        +----------------------+--------------------------------------------------------------------------------+
        | LaplacianEigenmaps_  | Detecting communities from multiple async label propagation on the graph       |
        +----------------------+--------------------------------------------------------------------------------+
        | LINE_                | Large-scale information network embedding                                      |
        +----------------------+--------------------------------------------------------------------------------+
        | LLE_                 | Locally Linear Embedding                                                       |
        +----------------------+--------------------------------------------------------------------------------+

        .. _OpenNE: https://github.com/thunlp/OpenNE/
        .. _GraphFactorization: https://github.com/thunlp/OpenNE/blob/master/src/openne/gf.py
        .. _GraRep: https://github.com/thunlp/OpenNE/blob/master/src/openne/grarep.py
        .. _HOPE: https://github.com/thunlp/OpenNE/blob/master/src/openne/hope.py
        .. _LaplacianEigenmaps: https://github.com/thunlp/OpenNE/blob/master/src/openne/lap.py
        .. _LINE: https://github.com/thunlp/OpenNE/blob/master/src/openne/line.py
        .. _LLE: https://github.com/thunlp/OpenNE/blob/master/src/openne/lle.py


    dimension: int
        the dimension of the label embedding vectors
    aggregation_function: 'add', 'multiply', 'average' or Callable
        the function used to aggregate label vectors for all labels assigned to each of the samples
    normalize_weights: boolean
        whether to normalize weights in the label graph by the number of samples or not
    param_dict
        parameters passed to the embedder, don't use the dimension and graph parameters, this class will set them at fit

    If you use this classifier please cite the relevant embedding method paper
    and the label network embedding for multi-label classification paper:

    .. code :: bibtex

        @article{zhang2007ml,
          title={ML-KNN: A lazy learning approach to multi-label learning},
          author={Zhang, Min-Ling and Zhou, Zhi-Hua},
          journal={Pattern recognition},
          volume={40},
          number={7},
          pages={2038--2048},
          year={2007},
          publisher={Elsevier}
        }

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
        "GraphFactorization": (GraphFactorization, "rep_size"),
        "GraRep": (GraRep, "dim"),
        "HOPE": (HOPE, "d"),
        "LaplacianEigenmaps": (LaplacianEigenmaps, "rep_size"),
        "LINE": (LINE, "rep_size"),
        "LLE": (LLE, "d"),
    }

    _AGGREGATION_FUNCTIONS = {
        "add": np.add.reduce,
        "multiply": np.multiply.reduce,
        "average": lambda x: np.average(x, axis=0),
    }

    def __init__(
        self,
        graph_builder,
        embedding,
        dimension,
        aggregation_function,
        normalize_weights,
        param_dict=None,
    ):
        if embedding not in self._EMBEDDINGS:
            raise ValueError(
                "Embedding must be one of {}".format(", ".join(self._EMBEDDINGS.keys()))
            )

        if aggregation_function in self._AGGREGATION_FUNCTIONS:
            self.aggregation_function = self._AGGREGATION_FUNCTIONS[
                aggregation_function
            ]
        elif callable(aggregation_function):
            self.aggregation_function = aggregation_function
        else:
            raise ValueError(
                "Aggregation function must be callable or one of {}".format(
                    ", ".join(self._AGGREGATION_FUNCTIONS.keys())
                )
            )

        self.embedding = embedding
        self.param_dict = param_dict if param_dict is not None else {}
        self.dimension = dimension
        self.graph_builder = graph_builder
        self.normalize_weights = normalize_weights

    def fit(self, X, y):
        self.fit_transform(X, y)

    def fit_transform(self, X, y):
        tf.compat.v1.reset_default_graph()
        self._init_openne_graph(y)
        embedding_class, dimension_key = self._EMBEDDINGS[self.embedding]
        param_dict = copy(self.param_dict)
        param_dict["graph"] = self.graph_
        param_dict[dimension_key] = self.dimension
        self.embeddings_ = embedding_class(**param_dict)
        return X, self._embedd_y(y)

    def _init_openne_graph(self, y):
        self.graph_ = Graph()
        self.graph_.G = nx.DiGraph()
        for (src, dst), w in self.graph_builder.transform(y).items():
            self.graph_.G.add_edge(src, dst)
            self.graph_.G.add_edge(dst, src)
            if self.normalize_weights:
                w = float(w) / y.shape[0]
            self.graph_.G[src][dst]["weight"] = w
            self.graph_.G[dst][src]["weight"] = w
        self.graph_.encode_node()

    def _embedd_y(self, y):
        empty_vector = np.zeros(shape=self.dimension)
        if sp.issparse(y):
            return np.array(
                [
                    self.aggregation_function(
                        [self.embeddings_.vectors[node] for node in row]
                    )
                    if len(row) > 0
                    else empty_vector
                    for row in _iterate_over_sparse_matrix(y)
                ]
            ).astype("float64")

        return np.array(
            [
                self.aggregation_function(
                    [
                        self.embeddings_.vectors[node]
                        for node, v in enumerate(row)
                        if v > 0
                    ]
                )
                if len(row) > 0
                else empty_vector
                for row in (y.A if isinstance(y, np.matrix) else y)
            ]
        ).astype("float64")


def _iterate_over_sparse_matrix(y):
    for r in range(y.shape[0]):
        yield y[r, :].indices
