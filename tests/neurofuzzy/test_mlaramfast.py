# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.neurofuzzy.MLARAM"""

# Import modules
import unittest

# Import from package
from skmultilearn.neurofuzzy import MLARAM

# Import from test suite
from tests.testsuite.classifier_base import ClassifierBaseTest


class MLARAMTest(ClassifierBaseTest):
    """Test cases for MLARAM class"""

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = MLARAM()
        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = MLARAM()
        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
