import unittest

from skmultilearn.lazy.brknn import BinaryRelevanceKNN
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split
from sklearn.utils.estimator_checks import check_estimator

class BrkNNTest(ClassifierBaseTest):
    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        for 
        classifier = RakelD(classifier = self.get_labelpowerset_with_svc(), labelset_size = 3)

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')
        self.assertClassifierWorksWithSparsity(classifier, 'dense')

if __name__ == '__main__':
    unittest.main()