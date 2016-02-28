import unittest
import scipy.sparse as sp
from skmultilearn.cluster.base import LabelSpaceClustererBase, LabelCooccurenceClustererBase

class ClustererBaseTests(unittest.TestCase):

    def test_base_fit_predict_is_abstract(self):
        base_class = LabelSpaceClustererBase()
        self.assertRaises(NotImplementedError, base_class.fit_predict, None, None)

    def test_edge_map_works(self):
        test_data = sp.lil_matrix([[0, 1], [1, 0], [1, 1]])

        base_class = LabelCooccurenceClustererBase()
        edge_map = base_class.generate_coocurence_adjacency_matrix(test_data)
        
        self.assertEqual(edge_map, base_class.edge_map)
        self.assertEqual(test_data.shape[1], base_class.label_count)
        self.assertEqual(len(edge_map), 1)
        self.assertEqual(edge_map[(0,1)], 1)

if __name__ == '__main__':
    unittest.main()