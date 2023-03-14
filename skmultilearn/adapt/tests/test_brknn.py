import unittest

from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class BRkNNTest(ClassifierBaseTest):
    TEST_NEIGHBORS = 3

    def classifiers(self):
        return [
            BRkNNaClassifier(k=BRkNNTest.TEST_NEIGHBORS),
            BRkNNbClassifier(k=BRkNNTest.TEST_NEIGHBORS),
        ]

    def test_if_brknn_classification_works_on_sparse_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, "sparse")

    def test_if_meka_classification_works_on_dense_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, "dense")

    def test_if_works_with_cross_validation(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithCV(classifier)


if __name__ == "__main__":
    unittest.main()
