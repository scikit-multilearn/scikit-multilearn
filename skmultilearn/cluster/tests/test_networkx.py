import numpy as np
import scipy.sparse as sp
import unittest
import pytest
from sklearn.datasets import make_multilabel_classification
from skmultilearn.cluster import NetworkXLabelCooccurenceClusterer
from .test_base import supported_graphbuilder_generator

def get_networkx_clusterers():
    for graph in supported_graphbuilder_generator():
        yield NetworkXLabelCooccurenceClusterer(graph_builder=graph)

class NetworkXLabelCooccurenceClustererTests(unittest.TestCase):
    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        for clusterer in get_networkx_clusterers():
            partition = clusterer.fit_predict(X, y)
            self.assertIsInstance(partition, np.ndarray)
            for label in range(y.shape[1]):
                assert any(label in subset for subset in partition)
