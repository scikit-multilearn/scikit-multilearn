import numpy as np
import scipy.sparse as sp
import unittest
from sklearn.datasets import make_multilabel_classification
from skmultilearn.cluster import GraphToolCooccurenceClusterer
from .test_base import supported_graphbuilder_generator


class GraphtoolClustererBaseTests(unittest.TestCase):
    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(
            sparse=True, return_indicator='sparse')
        assert sp.issparse(y)

        for graph in supported_graphbuilder_generator():
            for allow_overlap in [True, False]:
                for use_degree_corr in [True, False, None]:
                    for model_selection_criterium in ['mean_field', 'bethe']:
                        for verbose in [True, False]:
                            clusterer = GraphToolCooccurenceClusterer(
                                graph_builder=graph,
                                allow_overlap=allow_overlap,
                                n_iters=2, n_init_iters=2,
                                use_degree_corr=use_degree_corr,
                                model_selection_criterium=model_selection_criterium,
                                verbose=verbose)
                            self.assertEqual(clusterer.allow_overlap, allow_overlap)
                            self.assertEqual(clusterer.n_iters, 2)
                            self.assertEqual(clusterer.n_init_iters, 2)
                            self.assertEqual(clusterer.model_selection_criterium, model_selection_criterium)
                            self.assertEqual(clusterer.verbose, verbose)

                            partition = clusterer.fit_predict(X, y)
                            self.assertIsInstance(partition, np.ndarray)
                            self.assertEquals(partition.shape[0], y.shape[1])

if __name__ == '__main__':
    unittest.main()
