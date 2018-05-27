import numpy as np
import scipy.sparse as sp
import unittest
from sklearn.cluster import KMeans
from sklearn.datasets import make_multilabel_classification
from skmultilearn.cluster import MatrixLabelSpaceClusterer

def get_matrix_clusterers(cluster_count=3):
    base_clusterers = [KMeans(cluster_count)]
    for base_clusterer in base_clusterers:
        yield MatrixLabelSpaceClusterer(base_clusterer, False)

class MatrixLabelSpaceClustererTests(unittest.TestCase):

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(sparse=True, return_indicator='sparse')
        assert sp.issparse(y)
        cluster_count = 3

        for clusterer in get_matrix_clusterers(cluster_count):
            partition = clusterer.fit_predict(X, y)
            self.assertIsInstance(partition, np.ndarray)
            for label in range(y.shape[1]):
                self.assertTrue(any(label in subset for subset in partition))


if __name__ == '__main__':
    unittest.main()
