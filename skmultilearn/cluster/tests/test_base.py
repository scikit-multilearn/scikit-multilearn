import unittest

import scipy.sparse as sp

from skmultilearn.cluster.base import LabelGraphClustererBase, GraphBuilderBase
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder


def supported_graphbuilder_generator():
    for weighted in [True, False]:
        for include_self_edges in [True, False]:
            normalize_cases = [False]
            if weighted and include_self_edges:
                normalize_cases.append(True)
            for normalize_self_edges in normalize_cases:
                yield LabelCooccurrenceGraphBuilder(
                    weighted=weighted,
                    include_self_edges=include_self_edges,
                    normalize_self_edges=normalize_self_edges,
                )


class ClustererBaseTests(unittest.TestCase):
    def test_clusterer_base_fit_predict_is_abstract(self):
        base_class = LabelGraphClustererBase(GraphBuilderBase())
        self.assertRaises(NotImplementedError, base_class.fit_predict, None, None)

    def test_graph_builder_base_transform_is_abstract(self):
        base_class = GraphBuilderBase()
        self.assertRaises(NotImplementedError, base_class.transform, None)

    def test_weight_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurrenceGraphBuilder, "not bool", True, True
        )

    def test_include_self_edges_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurrenceGraphBuilder, True, "not bool", True
        )

    def test_normalize_self_edges_is_not_bool_exception(self):
        self.assertRaises(
            ValueError, LabelCooccurrenceGraphBuilder, True, True, "not bool"
        )

    def test_self_edge_normalization_requires_self_edge_inclusion(self):
        self.assertRaises(ValueError, LabelCooccurrenceGraphBuilder, False, False, True)

    def test_self_edge_normalization_requires_weighted_network(self):
        self.assertRaises(ValueError, LabelCooccurrenceGraphBuilder, False, True, True)

    def test_edge_map_works_unweighted_non_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurrenceGraphBuilder(
            weighted=False, include_self_edges=False, normalize_self_edges=False
        )
        edge_map = base_class.transform(test_data)

        self.assertEqual(len(edge_map), 1)
        self.assertEqual(edge_map[(0, 1)], 1)
        self.assertNotIn((1, 1), edge_map)

    def test_edge_map_works_weighted_non_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurrenceGraphBuilder(
            weighted=True, include_self_edges=False, normalize_self_edges=False
        )
        edge_map = base_class.transform(test_data)

        self.assertEqual(len(edge_map), 1)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertNotIn((1, 1), edge_map)

    def test_edge_map_works_unweighted_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurrenceGraphBuilder(
            weighted=False, include_self_edges=True, normalize_self_edges=False
        )
        edge_map = base_class.transform(test_data)

        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 1)
        self.assertEqual(edge_map[(1, 1)], 1)

    def test_edge_map_works_weighted_self_edged_non_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurrenceGraphBuilder(
            weighted=True, include_self_edges=True, normalize_self_edges=False
        )
        edge_map = base_class.transform(test_data)

        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertEqual(edge_map[(1, 1)], 3)

    def test_edge_map_works_weighted_self_edged_normalized(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1], [1, 1]])

        base_class = LabelCooccurrenceGraphBuilder(
            weighted=True, include_self_edges=True, normalize_self_edges=True
        )
        edge_map = base_class.transform(test_data)

        self.assertEqual(len(edge_map), 3)
        self.assertEqual(edge_map[(0, 1)], 2)
        self.assertEqual(edge_map[(1, 1)], 1.5)


if __name__ == "__main__":
    unittest.main()
