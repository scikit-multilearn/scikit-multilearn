import unittest

from skmultilearn.neurofuzzy import MLARAM
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class MLARAMTest(ClassifierBaseTest):

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = MLARAM()

        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = MLARAM()

        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
