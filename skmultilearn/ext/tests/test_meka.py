import unittest
import os

from skmultilearn.ext import Meka, download_meka
from skmultilearn.tests.classifier_basetest import ClassifierBaseTest
from skmultilearn.ext.meka import SUPPORTED_VERSION
from skmultilearn.dataset import clear_data_home


class MekaTest(ClassifierBaseTest):
    def classifier(self):
        meka_classifier = "meka.classifiers.multilabel.BR"
        weka_classifier = "weka.classifiers.rules.ZeroR"
        return Meka(meka_classifier=meka_classifier, weka_classifier=weka_classifier)

    def test_if_meka_classification_works_on_sparse_input(self):
        self.assertClassifierWorksWithSparsity(self.classifier(), "sparse")

    def test_if_meka_classification_works_on_dense_input(self):
        self.assertClassifierWorksWithSparsity(self.classifier(), "dense")

    def test_if_works_with_cross_validation(self):
        self.assertClassifierWorksWithCV(self.classifier())

    def test_if_downloading_meka_works(self):
        clear_data_home()
        path = download_meka()
        self.assertTrue(
            os.path.exists(os.path.join(path, "meka-{}.jar".format(SUPPORTED_VERSION)))
        )
        clear_data_home()


if __name__ == "__main__":
    unittest.main()
