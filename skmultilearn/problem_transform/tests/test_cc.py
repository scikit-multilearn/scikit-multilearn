import unittest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from ..tests.example import EXAMPLE_X, EXAMPLE_y

from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class CCTest(ClassifierBaseTest):

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

    def test_if_order_is_set(self):
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True], order=None
        )

        classifier.fit(EXAMPLE_X, EXAMPLE_y)

        self.assertEqual(classifier._order(), list(range(EXAMPLE_y.shape[1])))

    def test_if_order_is_set(self):
        reversed_chain = list(reversed(range(EXAMPLE_y.shape[1])))
        classifier = ClassifierChain(
            classifier=GaussianNB(), require_dense=[True, True], order=reversed_chain
        )

        classifier.fit(EXAMPLE_X, EXAMPLE_y)

        self.assertEqual(classifier._order(), reversed_chain)


if __name__ == '__main__':
    unittest.main()
