import numpy as np
import scipy.sparse as sp

from skmultilearn.cluster import balancedkmeans
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class BalancedKMeansClustererTest(ClassifierBaseTest):
    def test_actually_works_on_proper_params(self):
        for X, y in self.get_multilabel_data_for_tests("sparse"):
            assert sp.issparse(y)

            balanced = balancedkmeans.BalancedKMeansClusterer(k=3, it=50)
            partition = balanced.fit_predict(X, y)
            self.assertIsInstance(partition, np.ndarray)
            for label in range(y.shape[1]):
                self.assertTrue(any(label in subset for subset in partition))
