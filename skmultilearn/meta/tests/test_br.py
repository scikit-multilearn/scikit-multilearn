import unittest

from skmultilearn.meta.br import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split
from sklearn.utils.estimator_checks import check_estimator

class BRTest(unittest.TestCase):
    def classifier_verification(self, base_classifier, indicator, require_dense):
        classifier = BinaryRelevance(classifier = base_classifier, require_dense = require_dense)
        feed_sparse = indicator == 'sparse'
        X, y = make_multilabel_classification(sparse = feed_sparse, return_indicator = indicator)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier.fit(X_train, y_train)
        result = classifier.predict(X_test)

        self.assertEqual(result.shape, y_test.shape)

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        self.classifier_verification(SVC(), 'sparse', [False, True])

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        self.classifier_verification(SVC(), 'dense', [False, True])

    def test_if_sparse_classification_works_on_dense_base_classifier(self):
        self.classifier_verification(GaussianNB(), 'sparse', True)

    def test_if_dense_classification_works_on_dense_base_classifier(self):
        self.classifier_verification(GaussianNB(), 'dense', True)        

if __name__ == '__main__':
    unittest.main()