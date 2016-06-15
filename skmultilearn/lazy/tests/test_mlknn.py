import unittest

from skmultilearn.lazy.mlknn import MLkNN
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class MLkNNTest(ClassifierBaseTest):
    TEST_NEIGHBORS = 3

    def classifiers(self):
        return [MLkNN(k = MLkNNTest.TEST_NEIGHBORS, s = 1.0)]

    def test_if_mlknn_classification_works_on_sparse_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_mlknn_classification_works_on_dense_input(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_mlknn_works_with_cross_validation(self):
        for classifier in self.classifiers():
            self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
