import numpy as np
import scipy.sparse as sp
import unittest
from sklearn.datasets import make_multilabel_classification
from skmultilearn.cluster.igraph import IGraphLabelCooccurenceClusterer


class ClustererBaseTests(unittest.TestCase):

    def test_unsupported_methods_raise_exception(self):
        assert 'foobar' not in IGraphLabelCooccurenceClusterer.METHODS
        self.assertRaises(
            ValueError, IGraphLabelCooccurenceClusterer, 'foobar', False)

    def test_weight_is_not_bool_exception(self):
        method = IGraphLabelCooccurenceClusterer.METHODS.keys()[0]
        self.assertRaises(
            ValueError, IGraphLabelCooccurenceClusterer, method, 'not bool')

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        for method in IGraphLabelCooccurenceClusterer.METHODS.keys():
            for weighted in [True, False]:
                clusterer = IGraphLabelCooccurenceClusterer(method, weighted)
                self.assertEqual(clusterer.method, method)
                self.assertEqual(clusterer.is_weighted, weighted)
                partition = clusterer.fit_predict(X, y)
                self.assertIsInstance(partition, np.ndarray)


if __name__ == '__main__':
    unittest.main()
