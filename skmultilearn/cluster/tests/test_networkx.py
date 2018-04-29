import numpy as np
import scipy.sparse as sp
import unittest
from sklearn.datasets import make_multilabel_classification
from skmultilearn.cluster import NetworkXLabelCooccurenceClusterer
from .test_base import supported_graphbuilder_generator


class NetworkXLabelCooccurenceClustererTests(unittest.TestCase):
    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        for graph in supported_graphbuilder_generator():
            clusterer = NetworkXLabelCooccurenceClusterer(graph_builder=graph)

            partition = clusterer.fit_predict(X, y)
            self.assertIsInstance(partition, np.ndarray)
            self.assertEquals(partition.shape[0], y.shape[1])

if __name__ == '__main__':
    unittest.main()
