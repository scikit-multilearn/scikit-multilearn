import unittest
import sys

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from skmultilearn.cluster.tests.test_networkx import get_networkx_clusterers
from skmultilearn.cluster.tests.test_matrix import get_matrix_clusterers

if (sys.platform != "win32") and (
    sys.platform != "darwin" and sys.version_info[0] == 2
):
    from skmultilearn.cluster.tests.test_graphtool import get_graphtool_partitioners
    from skmultilearn.cluster.tests.test_igraph import get_igraph_clusterers


def generate_all_label_space_clusterers():
    for clusterer in get_networkx_clusterers():
        yield clusterer

    for clusterer in get_matrix_clusterers():
        yield clusterer

    if (sys.platform != "win32") and (
        sys.platform != "darwin" and sys.version_info[0] == 2
    ):
        for clusterer, _ in get_igraph_clusterers():
            yield clusterer

        for clusterer in get_graphtool_partitioners():
            yield clusterer


class LabelSpacePartitioningClassifierTest(ClassifierBaseTest):
    def get_labelpowerset_with_svc(self):
        return LabelPowerset(
            classifier=SVC(probability=True), require_dense=[False, True]
        )

    def get_labelpowerset_with_nb(self):
        return LabelPowerset(classifier=GaussianNB(), require_dense=[True, True])

    def get_classifier(self, base_classifier):
        for clusterer in generate_all_label_space_clusterers():
            yield LabelSpacePartitioningClassifier(
                classifier=base_classifier, clusterer=clusterer
            )

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        for classifier in self.get_classifier(self.get_labelpowerset_with_svc()):
            self.assertClassifierWorksWithSparsity(classifier, "sparse")
            self.assertClassifierPredictsProbabilities(classifier, "sparse")

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        for classifier in self.get_classifier(self.get_labelpowerset_with_svc()):
            self.assertClassifierWorksWithSparsity(classifier, "dense")
            self.assertClassifierPredictsProbabilities(classifier, "dense")

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        for classifier in self.get_classifier(self.get_labelpowerset_with_nb()):
            self.assertClassifierWorksWithSparsity(classifier, "sparse")
            self.assertClassifierPredictsProbabilities(classifier, "sparse")

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        for classifier in self.get_classifier(self.get_labelpowerset_with_nb()):
            self.assertClassifierWorksWithSparsity(classifier, "dense")
            self.assertClassifierPredictsProbabilities(classifier, "dense")

    def test_if_works_with_cross_validation(self):
        for classifier in self.get_classifier(self.get_labelpowerset_with_nb()):
            self.assertClassifierWorksWithCV(classifier)


if __name__ == "__main__":
    unittest.main()
