# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.ensemble.fixed module"""

import unittest
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Import from package
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import LabelPowerset

# Import from testsuite
from tests.testsuite.classifier_base import ClassifierBaseTest

# Import from package
try:
    # Try importing graph_tools
    import graph_tool.all as gt
except ImportError:
    # Set check_env = True
    check_env = True
else:
    from skmultilearn.cluster import IGraphLabelCooccurenceClusterer

@unittest.skipIf(check_env, 'Graphtool not found. Skipping all tests')
class LabelSpacePartitioningClassifierTest(ClassifierBaseTest):
    """Test cases for FixedLabelPartitionClassifier class"""

    def setUp(self):
        """Set-up test fixtures"""
        self.lp_svc = LabelPowerset(classifier=SVC(), 
                                    require_dense=[False, True])
        self.lp_nb = LabelPowerset(classifier=GaussianNB(),
                                    require_dense=[True,True])

    def get_classifier(self, base_classifier):
        clusterer = IGraphLabelCooccurenceClusterer('fastgreedy', False, False)
        return LabelSpacePartitioningClassifier(classifier=base_classifier, clusterer=clusterer)

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_classifier(self.lp_svc)
        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_classifier(self.lp_svc)
        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        classifier = self.get_classifier(self.lp_nb)
        self.assertClassifierWorksWithSparsity(classifier, 'sparse')

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        classifier = self.get_classifier(self.lp_nb)
        self.assertClassifierWorksWithSparsity(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = self.get_classifier(self.lp_nb)
        self.assertClassifierWorksWithCV(classifier)

if __name__ == '__main__':
    unittest.main()
