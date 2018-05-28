import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_multilabel_classification

from skmultilearn.cluster import RandomLabelSpaceClusterer

TEST_PARTITION_SIZE = 2
TEST_PARTITION_COUNT = 2


def get_networkx_clusterers():
    for overlap in [True, False]:
        if overlap:
            yield RandomLabelSpaceClusterer(partition_count=TEST_PARTITION_COUNT + 1,
                                            partition_size=TEST_PARTITION_SIZE,
                                            allow_overlap=overlap)
        else:
            yield RandomLabelSpaceClusterer(partition_count=TEST_PARTITION_COUNT, partition_size=TEST_PARTITION_SIZE,
                                            allow_overlap=overlap)


class RandomClustererTests(unittest.TestCase):
    def test_actually_works_on_proper_params(self):
        X, y = make_multilabel_classification(sparse=True, return_indicator='sparse',
                                              n_classes=TEST_PARTITION_COUNT * TEST_PARTITION_SIZE)
        assert sp.issparse(y)

        for clusterer in get_networkx_clusterers():
            partition = clusterer.fit_predict(X, y)
            self.assertIsInstance(partition, np.ndarray)
            for label in range(y.shape[1]):
                assert any(label in subset for subset in partition)
