import unittest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skmultilearn.ensemble import FixedLabelPartitionClassifier
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class LabelSpacePartitioningClassifierTest(ClassifierBaseTest):

    def get_labelpowerset_with_svc(self):
        return LabelPowerset(classifier=SVC(), require_dense=[False, True])

    def get_labelpowerset_with_nb(self):
        return LabelPowerset(classifier=GaussianNB(), require_dense=[True, True])

    def get_classifier(self, base_classifier):
        partition = [[0, 1, 2], [3, 4]]
        return FixedLabelPartitionClassifier(partition=partition, classifier=base_classifier)

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_classifier(self.get_labelpowerset_with_svc())

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_classifier(self.get_labelpowerset_with_svc())

        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = self.get_classifier(self.get_labelpowerset_with_nb())

        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = self.get_classifier(self.get_labelpowerset_with_nb())

        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = self.get_classifier(self.get_labelpowerset_with_nb())

        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
