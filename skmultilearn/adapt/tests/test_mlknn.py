import unittest

from skmultilearn.adapt import MLkNN
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class MLkNNTest(ClassifierBaseTest):
    TEST_NEIGHBORS = 3
    N_JOBS = -1

    def classifiers(self):
        return [MLkNN(k=MLkNNTest.TEST_NEIGHBORS, s=1.0)]

    def test_if_mlknn_classification_works_on_sparse_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, "sparse")
            self.assertClassifierPredictsProbabilities(classifier, "sparse")

    def test_if_mlknn_classification_works_on_dense_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, "dense")
            self.assertClassifierPredictsProbabilities(classifier, "dense")

    def test_if_mlknn_works_with_cross_validation(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithCV(classifier)

    def test_mlknn_instantiates_nearest_neighbor(self):
        classifier = MLkNN(k=MLkNNTest.TEST_NEIGHBORS, n_jobs=MLkNNTest.N_JOBS)
        self.assertIsNotNone(classifier.knn_)

    def test_mlknn_supports_parallelization(self):
        classifier = MLkNN(k=MLkNNTest.TEST_NEIGHBORS, n_jobs=MLkNNTest.N_JOBS)
        knn_params = classifier.knn_.get_params()

        self.assertEqual(knn_params["n_jobs"], MLkNNTest.N_JOBS)


if __name__ == "__main__":
    unittest.main()
