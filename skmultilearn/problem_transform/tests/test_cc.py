import unittest

from skmultilearn.problem_transform.cc import ClassifierChain
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split
from sklearn.utils.estimator_checks import check_estimator


class CCTest(ClassifierBaseTest):

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=SVC(), require_dense=[False, True])

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=SVC(), require_dense=[False, True])

        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
