# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.cluster module"""

# Import modules
import unittest
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_multilabel_classification

# Import from package
try:
    # Try importing graph_tools
    import graph_tool.all as gt
except ImportError:
    # Set check_env = True
    check_env = True
else:
    from skmultilearn.cluster import IGraphLabelCooccurenceClusterer

@unittest.skipIf(check_env, 'Graphtool not found. Skipping all tests')
class ClustererBaseTests(unittest.TestCase):
    """Test cases for IGraphLabelCooccurenceClusterer class"""

    def test_unsupported_methods_raise_exception(self):
        assert 'foobar' not in IGraphLabelCooccurenceClusterer.METHODS
        self.assertRaises(
            ValueError, IGraphLabelCooccurenceClusterer, 'foobar', False)

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        for method in list(IGraphLabelCooccurenceClusterer.METHODS.keys()):
            for weighted in [True, False]:
                for include_self_edges in [True, False]:
                    clusterer = IGraphLabelCooccurenceClusterer(method, weighted=weighted,
                                                                include_self_edges=include_self_edges)
                    self.assertEqual(clusterer.method, method)
                    self.assertEqual(clusterer.is_weighted, weighted)
                    partition = clusterer.fit_predict(X, y)
                    self.assertIsInstance(partition, np.ndarray)


if __name__ == '__main__':
    unittest.main()
