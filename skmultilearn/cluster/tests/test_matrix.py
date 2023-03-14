import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from skmultilearn.cluster import MatrixLabelSpaceClusterer
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


def get_matrix_clusterers(cluster_count=3):
    base_clusterers = [KMeans(cluster_count)]
    for base_clusterer in base_clusterers:
        yield MatrixLabelSpaceClusterer(base_clusterer, False)


class MatrixLabelSpaceClustererTests(ClassifierBaseTest):
    def test_actually_works_on_proper_params(self):
        for X, y in self.get_multilabel_data_for_tests("sparse"):
            assert sp.issparse(y)
            cluster_count = 3

            for clusterer in get_matrix_clusterers(cluster_count):
                partition = clusterer.fit_predict(X, y)
                self.assertIsInstance(partition, np.ndarray)
                for label in range(y.shape[1]):
                    self.assertTrue(any(label in subset for subset in partition))
