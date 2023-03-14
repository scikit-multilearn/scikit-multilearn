import unittest

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skmultilearn.ensemble import RakelO
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest

TEST_MODEL_COUNT = 3
TEST_LABELSET_SIZE = 2


class RakelOTest(ClassifierBaseTest):
    def get_rakeld_with_svc(self):
        return RakelO(
            base_classifier=SVC(),
            base_classifier_require_dense=[False, True],
            labelset_size=TEST_LABELSET_SIZE,
            model_count=TEST_MODEL_COUNT,
        )

    def get_rakeld_with_nb(self):
        return RakelO(
            base_classifier=GaussianNB(),
            base_classifier_require_dense=[True, True],
            labelset_size=TEST_LABELSET_SIZE,
            model_count=TEST_MODEL_COUNT,
        )

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_rakeld_with_svc()
        self.assertClassifierWorksWithSparsity(classifier, "sparse")

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_rakeld_with_svc()
        self.assertClassifierWorksWithSparsity(classifier, "dense")

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = self.get_rakeld_with_nb()
        self.assertClassifierWorksWithSparsity(classifier, "sparse")

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = self.get_rakeld_with_nb()
        self.assertClassifierWorksWithSparsity(classifier, "dense")

    def test_if_works_with_cross_validation(self):
        classifier = self.get_rakeld_with_nb()

        self.assertClassifierWorksWithCV(classifier)


if __name__ == "__main__":
    unittest.main()
