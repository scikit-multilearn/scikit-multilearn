import unittest

import scipy.sparse as sp

from skmultilearn.cluster import LabelSpaceClustererBase, LabelCooccurenceClustererBase


class ClustererBaseTests(unittest.TestCase):

    def test_base_fit_predict_is_abstract(self):
        base_class = LabelSpaceClustererBase()
        self.assertRaises(NotImplementedError,
                          base_class.fit_predict, None, None)

    def test_weight_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurenceClustererBase, 'not bool', True, True)

    def test_include_self_edges_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurenceClustererBase, True, 'not bool', True)

    def test_normalize_self_edges_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurenceClustererBase, True, True, 'not bool')

    def test_self_edge_normalization_requires_self_edge_inclusion(self):
        self.assertRaises(
            ValueError, LabelCooccurenceClustererBase, False, False, True)

    def test_self_edge_normalization_requires_weighted_network(self):
        self.assertRaises(
            ValueError, LabelCooccurenceClustererBase, False, True, True)

    def test_edge_map_works_unweighted_non_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=False, include_self_edges=False, normalize_self_edges=False)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 1)
        self.assertEqual(edge_map[(0, 1)], 1)
        self.assertNotIn((1, 1), edge_map)

    def test_edge_map_works_weighted_non_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=True, include_self_edges=False, normalize_self_edges=False)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 1)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertNotIn((1, 1), edge_map)

    def test_edge_map_works_unweighted_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=False, include_self_edges=True, normalize_self_edges=False)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 1)
        self.assertEqual(edge_map[(1, 1)], 1)

    def test_edge_map_works_weighted_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=True, include_self_edges=True, normalize_self_edges=False)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertEqual(edge_map[(1, 1)], 3)

    def test_edge_map_works_weighted_self_edged_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurenceClustererBase(weighted=True, include_self_edges=True, normalize_self_edges=True)
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)

        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertEqual(edge_map[(1, 1)], 1.5)


if __name__ == '__main__':
    unittest.main()
