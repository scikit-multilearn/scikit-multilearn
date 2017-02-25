import numpy as np
import scipy.sparse as sp
import unittest
from sklearn.cluster import KMeans
from sklearn.datasets import make_multilabel_classification
from skmultilearn.cluster import MatrixLabelSpaceClusterer

class GraphtoolClustererBaseTests(unittest.TestCase):

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        base_clusterer = KMeans(3)

        clusterer = MatrixLabelSpaceClusterer(base_clusterer, False)

        partition = clusterer.fit_predict(X, y)
        self.assertIsInstance(partition, np.ndarray)
        
if __name__ == '__main__':
    unittest.main()
