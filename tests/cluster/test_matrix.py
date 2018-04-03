# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.cluster module"""

# Import modules
import unittest
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.datasets import make_multilabel_classification


# Import from package
try:
    # Try importing graph_tools
    import graph_tool.all as gt
except ImportError:
    # Set check_env = True
    check_env = True
else:
    from skmultilearn.cluster import MatrixLabelSpaceClusterer

@unittest.skipIf(check_env, 'Graphtool not found. Skipping all tests')
class GraphtoolClustererBaseTests(unittest.TestCase):
    """Test cases for MatrixLabelSpaceClusterer"""

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
