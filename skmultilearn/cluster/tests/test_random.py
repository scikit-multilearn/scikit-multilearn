import numpy as np
import scipy.sparse as sp

from skmultilearn.cluster import RandomLabelSpaceClusterer
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest

TEST_PARTITION_SIZE = 2
TEST_PARTITION_COUNT = 3


def get_random_clusterers():
    for overlap in [True, False]:
        if overlap:
            yield RandomLabelSpaceClusterer(
                cluster_count=TEST_PARTITION_COUNT * TEST_PARTITION_SIZE,
                cluster_size=TEST_PARTITION_SIZE,
                allow_overlap=overlap,
            )
        else:
            yield RandomLabelSpaceClusterer(
                cluster_count=TEST_PARTITION_COUNT,
                cluster_size=TEST_PARTITION_SIZE,
                allow_overlap=overlap,
            )


class RandomClustererTests(ClassifierBaseTest):
    def test_actually_works_on_proper_params(self):
        for X, y in self.get_multilabel_data_for_tests("sparse"):
            assert sp.issparse(y)

            for clusterer in get_random_clusterers():
                partition = clusterer.fit_predict(X, y)
                self.assertIsInstance(partition, np.ndarray)
                for label in range(y.shape[1]):
                    assert any(label in subset for subset in partition)
