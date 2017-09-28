# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.adapt module"""

# Import modules
import unittest

# Import from package
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier
from .base.classifier_base_tests import ClassifierBaseTest


class BRkNNTest(ClassifierBaseTest):
    """Test cases for skmultilear.adapt.BrkNNaClassifier and BrkNNbClassifier"""

    def setUp(self):
        """Set-up test fixtures"""
        self.neighbors = 3
        self.classifiers = [
            BRkNNaClassifier(k=self.neighbors),
            BRkNNbClassifier(k=self.neighbors)
        ]

    def test_if_brknn_classification_works_on_sparse_input(self):
        for classifier in self.classifiers:
            self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_meka_classification_works_on_dense_input(self):
        for classifier in self.classifiers:
            self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        for classifier in self.classifiers:
            self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
