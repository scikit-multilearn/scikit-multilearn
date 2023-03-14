import unittest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class LPTest(ClassifierBaseTest):
    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = LabelPowerset(
            classifier=SVC(probability=True), require_dense=[False, True]
        )

        self.assertClassifierWorksWithSparsity(classifier, "sparse")
        self.assertClassifierPredictsProbabilities(classifier, "sparse")

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = LabelPowerset(
            classifier=SVC(probability=True), require_dense=[False, True]
        )

        self.assertClassifierWorksWithSparsity(classifier, "dense")
        self.assertClassifierPredictsProbabilities(classifier, "dense")

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = LabelPowerset(classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithSparsity(classifier, "sparse")
        self.assertClassifierPredictsProbabilities(classifier, "sparse")

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = LabelPowerset(classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithSparsity(classifier, "dense")
        self.assertClassifierPredictsProbabilities(classifier, "dense")

    def test_if_works_with_cross_validation(self):
        classifier = LabelPowerset(classifier=GaussianNB(), require_dense=[True, True])

        self.assertClassifierWorksWithCV(classifier)


if __name__ == "__main__":
    unittest.main()
