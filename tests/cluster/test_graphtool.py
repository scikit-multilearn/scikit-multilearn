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
    from skmultilearn.cluster import GraphToolCooccurenceClusterer

@unittest.skipIf(check_env, 'Graphtool not found. Skipping all tests')
class GraphtoolClustererBaseTests(unittest.TestCase):

    def test_allow_overlap_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, GraphToolCooccurenceClusterer, True, 'not bool')

    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        for allow_overlap in [True, False]:
            for weighted in [True, False]:
                for include_self_edges in [True, False]:
                    for use_degree_corr in [True, False, None]:
                        for model_selection_criterium in ['mean_field', 'bethe']:
                            for verbose in [True, False]:
                                clusterer = GraphToolCooccurenceClusterer(
                                    weighted=weighted, allow_overlap=allow_overlap, 
                                    include_self_edges=include_self_edges,
                                    n_iters=2, n_init_iters=2, 
                                    use_degree_corr=use_degree_corr,
                                    model_selection_criterium=model_selection_criterium,
                                    verbose=verbose)
                                self.assertEqual(clusterer.allow_overlap, allow_overlap)
                                self.assertEqual(clusterer.is_weighted, weighted)
                                self.assertEqual(clusterer.include_self_edges, include_self_edges)
                                self.assertEqual(clusterer.n_iters, 2)
                                self.assertEqual(clusterer.n_init_iters, 2)
                                self.assertEqual(clusterer.model_selection_criterium, model_selection_criterium)
                                self.assertEqual(clusterer.verbose, verbose)

                                partition = clusterer.fit_predict(X, y)
                                self.assertIsInstance(partition, np.ndarray)

if __name__ == '__main__':
    unittest.main()
