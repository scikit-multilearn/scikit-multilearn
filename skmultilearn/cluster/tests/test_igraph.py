import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_multilabel_classification

from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
from .test_base import supported_graphbuilder_generator


class IGraphClustererBaseTests(unittest.TestCase):

    def test_unsupported_methods_raise_exception(self):
        assert 'foobar' not in IGraphLabelCooccurenceClusterer.METHODS
        self.assertRaises(
            ValueError, IGraphLabelCooccurenceClusterer, 'foobar', False)

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        for graph in supported_graphbuilder_generator():
            for method in IGraphLabelCooccurenceClusterer.METHODS.keys():
                clusterer = IGraphLabelCooccurenceClusterer(graph_builder=graph, method=method)
                self.assertEqual(clusterer.method, method)
                partition = clusterer.fit_predict(X, y)
                self.assertIsInstance(partition, np.ndarray)
                self.assertEquals(partition.shape[0], y.shape[1])

if __name__ == '__main__':
    unittest.main()
