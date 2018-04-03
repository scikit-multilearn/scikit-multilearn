# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.problem_transform.cc module"""

# Import modules
import unittest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Import from package
from skmultilearn.problem_transform import ClassifierChain

# Import from test suite
from tests.testsuite.classifier_base import ClassifierBaseTest


class CCTest(ClassifierBaseTest):
    """Test cases for ClassifierChain class"""

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=SVC(probability=True), require_dense=[False, True])

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')
        self.assertClassifierPredictsProbabilities(classifier, 'sparse')

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=SVC(probability=True), require_dense=[False, True])

        self.assertClassifierWorksWithSparsity(classifier, 'dense')
        self.assertClassifierPredictsProbabilities(classifier, 'dense')

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')
        self.assertClassifierPredictsProbabilities(classifier, 'sparse')

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithSparsity(classifier, 'dense')
        self.assertClassifierPredictsProbabilities(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
