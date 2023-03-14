import numpy as np
import scipy.sparse as sp
import sys

if sys.platform != "win32":
    from skmultilearn.cluster import IGraphLabelGraphClusterer
    from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
    from .test_base import supported_graphbuilder_generator

    def get_igraph_clusterers():
        for graph in supported_graphbuilder_generator():
            for method in IGraphLabelGraphClusterer._METHODS.keys():
                yield IGraphLabelGraphClusterer(
                    graph_builder=graph, method=method
                ), method

    class IGraphClustererBaseTests(ClassifierBaseTest):
        def test_unsupported_methods_raise_exception(self):
            assert "foobar" not in IGraphLabelGraphClusterer._METHODS
            self.assertRaises(ValueError, IGraphLabelGraphClusterer, "foobar", False)

        def test_actually_works_on_proper_params(self):
            for X, y in self.get_multilabel_data_for_tests("sparse"):
                assert sp.issparse(y)
                for clusterer, method in get_igraph_clusterers():
                    self.assertEqual(clusterer.method, method)
                    partition = clusterer.fit_predict(X, y)
                    self.assertIsInstance(partition, np.ndarray)
                    for label in range(y.shape[1]):
                        assert any(label in subset for subset in partition)
