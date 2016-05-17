import unittest

from sklearn.datasets import make_multilabel_classification
from sklearn import cross_validation
from sklearn.utils.estimator_checks import check_estimator

class ClassifierBaseTest(unittest.TestCase):
    def assertClassifierWorksWithSparsity(self, classifier, sparsity_indicator = 'sparse'):
        feed_sparse = sparsity_indicator == 'sparse'
        X, y = make_multilabel_classification(sparse = feed_sparse, return_indicator = sparsity_indicator)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)
        classifier.fit(X_train, y_train)
        result = classifier.predict(X_test)

        self.assertEqual(result.shape, y_test.shape)

    def assertClassifierWorksWithCV(self, classifier):
        # all the nice stuff is tested here - whether the classifier is clonable, etc.
        X, y = make_multilabel_classification()
        n_iterations = 3
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=n_iterations, test_size=0.3, random_state=0)
        scores = cross_validation.cross_val_score(classifier, X, y, cv=cv, scoring='f1_macro')

        self.assertEqual(len(scores), n_iterations)



if __name__ == '__main__':
    unittest.main()