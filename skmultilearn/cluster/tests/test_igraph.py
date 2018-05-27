import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_multilabel_classification

from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
from .test_base import supported_graphbuilder_generator


def get_igraph_clusterers():
    for graph in supported_graphbuilder_generator():
        for method in IGraphLabelCooccurenceClusterer.METHODS.keys():
            yield IGraphLabelCooccurenceClusterer(graph_builder=graph, method=method), method


class IGraphClustererBaseTests(unittest.TestCase):

    def test_unsupported_methods_raise_exception(self):
        assert 'foobar' not in IGraphLabelCooccurenceClusterer.METHODS
        self.assertRaises(
            ValueError, IGraphLabelCooccurenceClusterer, 'foobar', False)

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)
        for clusterer, method in get_igraph_clusterers():
            self.assertEqual(clusterer.method, method)
            partition = clusterer.fit_predict(X, y)
            self.assertIsInstance(partition, np.ndarray)
            for label in range(y.shape[1]):
                assert any(label in subset for subset in partition)
