from builtins import range
import unittest
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from skmultilearn.ensemble.partition import LabelSpacePartitioningClassifier
from skmultilearn.cluster import RandomLabelSpaceClusterer
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest


class MLClassifierBaseTests(ClassifierBaseTest):
    def test_model_selection_works(self):
        for x, y in self.get_multilabel_data_for_tests("dense"):
            parameters = {
                "classifier": [LabelPowerset(), BinaryRelevance()],
                "clusterer": [RandomLabelSpaceClusterer(None, None, False)],
                "clusterer__cluster_size": list(range(2, 3)),
                "clusterer__cluster_count": [3],
                "clusterer__allow_overlap": [False],
                "classifier__classifier": [MultinomialNB()],
                "classifier__classifier__alpha": [0.7, 1.0],
            }

            clf = GridSearchCV(
                LabelSpacePartitioningClassifier(), parameters, scoring="f1_macro"
            )
            clf.fit(x, y)

            for p in list(parameters.keys()):
                self.assertIn(p, clf.best_params_)

            self.assertIsNotNone(clf.best_score_)
