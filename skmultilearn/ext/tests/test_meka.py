import unittest

from skmultilearn.ext.meka import Meka
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class MekaTest(ClassifierBaseTest):

    def classifier(self):
        meka_classifier = "meka.classifiers.multilabel.BR"
        weka_classifier = "weka.classifiers.rules.ZeroR"
        return Meka(meka_classifier=meka_classifier, weka_classifier=weka_classifier)

    def test_if_meka_classification_works_on_sparse_input(self):
        self.assertClassifierWorksWithSparsity(self.classifier(), 'sparse')

    def test_if_meka_classification_works_on_dense_input(self):
        self.assertClassifierWorksWithSparsity(self.classifier(), 'dense')

    def test_if_works_with_cross_validation(self):
        self.assertClassifierWorksWithCV(self.classifier())

if __name__ == '__main__':
    unittest.main()
