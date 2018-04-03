# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.ext module"""

# Import modules
import os
import unittest
import argparse

# Import from package
from skmultilearn.ext import Meka

# Import from test suite
from tests.testsuite.classifier_base import ClassifierBaseTest

# Check if MEKA_CLASPATH exists in the environment variables
check_env = lambda : os.environ.get('MEKA_CLASSPATH')

@unittest.skipIf(check_env()==None, 'No MEKA_CLASSPATH found. Skipping MEKA extension test')
class MekaTest(ClassifierBaseTest):
    """Test cases for Meka"""

    def setUp(self):
        """Set-up test fixtures"""
        meka_classifier = "meka.classifiers.multilabel.BR"
        weka_classifier = "weka.classifiers.rules.ZeroR"
        self.classifier = Meka(meka_classifier=meka_classifier, 
                            weka_classifier=weka_classifier)

    def test_if_meka_classification_works_on_sparse_input(self):
        self.assertClassifierWorksWithSparsity(self.classifier, 'sparse')

    def test_if_meka_classification_works_on_dense_input(self):
        self.assertClassifierWorksWithSparsity(self.classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        self.assertClassifierWorksWithCV(self.classifier)

if __name__ == '__main__':
    unittest.main()
