# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.ensemble.rakeld module"""

# Import modules
import unittest
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Import from package
from skmultilearn.ensemble import RakelD
from skmultilearn.problem_transform import LabelPowerset

# Import from test suite
from tests.testsuite.classifier_base import ClassifierBaseTest

class RakelDTest(ClassifierBaseTest):
    """Test cases for RakelD class"""

    def setUp(self):
        """Set-up test fixtures"""
        self.lp_svc = LabelPowerset(classifier=SVC(), 
                                    require_dense=[False, True])
        self.lp_nb = LabelPowerset(classifier=GaussianNB(),
                                    require_dense=[True,True])

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = RakelD(classifier=self.lp_svc, labelset_size=3)
        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = RakelD(classifier=self.lp_svc, labelset_size=3)
        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = RakelD(classifier=self.lp_nb, labelset_size=3)
        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = RakelD(classifier=self.lp_nb, labelset_size=3)
        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = RakelD(classifier=self.lp_nb, labelset_size=3)
        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
