import unittest

from skmultilearn.meta.br import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split
from sklearn.utils.estimator_checks import check_estimator

class BRTest(unittest.TestCase):
    def test_if_sparse_classification_works(self):
        classifier = BinaryRelevance(DecisionTreeClassifier())
        X, y = make_multilabel_classification(sparse = True, return_indicator = 'sparse')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier.fit(X_train, y_train)
        result = classifier.predict(X_test)

        self.assertEqual(result.shape, y_test.shape)

    def test_if_dense_classification_works(self):
        classifier = BinaryRelevance(DecisionTreeClassifier())
        X, y = make_multilabel_classification(sparse = False, return_indicator = 'dense')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        classifier.fit(X_train, y_train)
        result = classifier.predict(X_test)
        
        self.assertEqual(result.shape, y_test.shape)

if __name__ == '__main__':
    unittest.main()