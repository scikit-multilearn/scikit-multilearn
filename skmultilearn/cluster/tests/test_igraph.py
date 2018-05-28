import numpy as np
import scipy.sparse as sp

from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from .test_base import supported_graphbuilder_generator


def get_igraph_clusterers():
    for graph in supported_graphbuilder_generator():
        for method in IGraphLabelCooccurenceClusterer.METHODS.keys():
            yield IGraphLabelCooccurenceClusterer(graph_builder=graph, method=method), method


class IGraphClustererBaseTests(ClassifierBaseTest):

    def test_unsupported_methods_raise_exception(self):
        assert 'foobar' not in IGraphLabelCooccurenceClusterer.METHODS
        self.assertRaises(
            ValueError, IGraphLabelCooccurenceClusterer, 'foobar', False)

    def test_actually_works_on_proper_params(self):
        for X, y in self.get_multilabel_data_for_tests('sparse'):
            assert sp.issparse(y)
            for clusterer, method in get_igraph_clusterers():
                self.assertEqual(clusterer.method, method)
                partition = clusterer.fit_predict(X, y)
                self.assertIsInstance(partition, np.ndarray)
                for label in range(y.shape[1]):
                    assert any(label in subset for subset in partition)
