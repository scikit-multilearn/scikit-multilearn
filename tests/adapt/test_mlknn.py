# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.adapt module"""

# Import modules
import unittest

# Import from package
from skmultilearn.adapt import MLkNN

# Import from testsuite
from tests.testsuite.classifier_base import ClassifierBaseTest


class MLkNNTest(ClassifierBaseTest):
    """Test cases for MLkNN class"""

    def setUp(self):
        """Set-up test fixtures"""
        self.neighbors = 3
        self.classifiers = [
            MLkNN(k=self.neighbors, s = 1.0)
        ]

    def test_if_mlknn_classification_works_on_sparse_input(self):
        for classifier in self.classifiers:
            self.assertClassifierWorksWithSparsity(classifier, 'sparse')
            self.assertClassifierPredictsProbabilities(classifier, 'sparse')

    def test_if_mlknn_classification_works_on_dense_input(self):
        for classifier in self.classifiers:
            self.assertClassifierWorksWithSparsity(classifier, 'dense')
            self.assertClassifierPredictsProbabilities(classifier, 'dense')


    def test_if_mlknn_works_with_cross_validation(self):
        for classifier in self.classifiers:
            self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
