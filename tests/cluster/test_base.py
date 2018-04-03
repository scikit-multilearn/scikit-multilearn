# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.cluster module"""

# Import modules
import unittest
import scipy.sparse as sp

# Import from package
try:
    # Try importing graph_tools
    import graph_tool.all as gt
except ImportError:
    # Set check_env = True
    check_env = True
else:
    from skmultilearn.cluster import LabelSpaceClustererBase, LabelCooccurenceClustererBase

@unittest.skipIf(check_env, 'Graphtool not found. Skipping all tests')
class ClustererBaseTests(unittest.TestCase):
    """Test cases for LabelSpaceClustererBase and LabelCooccurenceClustererBase"""

    def test_base_fit_predict_is_abstract(self):
        base_class = LabelSpaceClustererBase()
        self.assertRaises(NotImplementedError,
                          base_class.fit_predict, None, None)

    def test_weight_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurenceClustererBase, 'not bool', True)

    def test_iclude_self_edges_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurenceClustererBase, True, 'not bool')

    def test_edge_map_works_unweighted_non_self_edged(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=False, include_self_edges=False)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 1)
        self.assertEqual(edge_map[(0, 1)], 1)
        self.assertNotIn((1, 1), edge_map)

    def test_edge_map_works_weighted_non_self_edged(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=True, include_self_edges=False)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 1)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertNotIn((1, 1), edge_map)

    def test_edge_map_works_unweighted_self_edged(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=False, include_self_edges=True)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 1)
        self.assertEqual(edge_map[(1, 1)], 1)

    def test_edge_map_works_weighted_self_edged(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=True, include_self_edges=True)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertEqual(edge_map[(1, 1)], 3)

if __name__ == '__main__':
    unittest.main()
