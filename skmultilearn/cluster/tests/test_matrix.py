import numpy as np
import scipy.sparse as sp
import unittest
from sklearn.cluster import KMeans
from sklearn.datasets import make_multilabel_classification
from skmultilearn.cluster import MatrixLabelSpaceClusterer

class MatrixLabelSpaceClustererTests(unittest.TestCase):

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(sparse=True, return_indicator='sparse')
        assert sp.issparse(y)
        cluster_count = 3
        base_clusterer = KMeans(cluster_count)

        clusterer = MatrixLabelSpaceClusterer(base_clusterer, False)

        partition = clusterer.fit_predict(X, y)
        self.assertIsInstance(partition, np.ndarray)
        self.assertEquals(partition.shape[0], y.shape[1])
        self.assertLess(max(partition), cluster_count)

if __name__ == '__main__':
    unittest.main()
