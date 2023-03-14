import scipy.sparse as sp
import numpy as np

from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


def get_fixed_clusterers():
    cluster_cases = [
        # non-overlapping
        [[1, 2], [0, 3, 4]],
        # overlapping
        [[1, 2, 3], [0, 3, 4]],
    ]
    for clusters in cluster_cases:
        clusters = np.array(clusters)
        yield clusters, FixedLabelSpaceClusterer(clusters=clusters)


class FixedLabelSpaceClustererTests(ClassifierBaseTest):
    def test_actually_works_on_proper_params(self):
        for X, y in self.get_multilabel_data_for_tests("sparse"):
            assert sp.issparse(y)

            for clusters, clusterer in get_fixed_clusterers():
                self.assertEqual(clusterer.clusters.tolist(), clusters.tolist())

                division = clusterer.fit_predict(X, y)
                self.assertIsInstance(division, np.ndarray)
                print(division, clusters)
                self.assertEqual(division.tolist(), clusters.tolist())
                for label in range(y.shape[1]):
                    self.assertTrue(any(label in subset for subset in division))
