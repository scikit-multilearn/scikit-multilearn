import numpy as np
import scipy.sparse as sp

from skmultilearn.cluster import NetworkXLabelGraphClusterer
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from .test_base import supported_graphbuilder_generator


def get_networkx_clusterers():
    for graph in supported_graphbuilder_generator():
        for method in ["louvain", "label_propagation"]:
            yield NetworkXLabelGraphClusterer(graph_builder=graph, method=method)


class NetworkXLabelCooccurenceClustererTests(ClassifierBaseTest):
    def test_actually_works_on_proper_params(self):
        for X, y in self.get_multilabel_data_for_tests("sparse"):
            assert sp.issparse(y)

            for clusterer in get_networkx_clusterers():
                partition = clusterer.fit_predict(X, y)
                self.assertIsInstance(partition, np.ndarray)
                for label in range(y.shape[1]):
                    assert any(label in subset for subset in partition)
