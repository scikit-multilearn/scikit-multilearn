import unittest
from sklearn import model_selection
import numpy as np
import scipy.sparse as sparse
from .example import EXAMPLE_X, EXAMPLE_y

class ClassifierBaseTest(unittest.TestCase):
    def get_multilabel_data_for_tests(self, sparsity_indicator):
        feed_sparse = sparsity_indicator == 'sparse'

        if feed_sparse:
            return [(sparse.csr_matrix(EXAMPLE_X), sparse.csr_matrix(EXAMPLE_y))]
        else:
            return [(np.matrix(EXAMPLE_X), np.matrix(EXAMPLE_y))]


    def assertClassifierWorksWithSparsity(self, classifier, sparsity_indicator='sparse'):
        for X,y in self.get_multilabel_data_for_tests(sparsity_indicator):
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
            classifier.fit(X_train, y_train)
            result = classifier.predict(X_test)
            self.assertEqual(result.shape, y_test.shape)

    def assertClassifierWorksWithCV(self, classifier):
        # all the nice stuff is tested here - whether the classifier is
        # clonable, etc.
        for X, y in self.get_multilabel_data_for_tests('dense'):
            n_iterations = 3
            cv = model_selection.ShuffleSplit(n_splits=n_iterations, test_size=0.5, random_state=0)

            scores = model_selection.cross_val_score(
                classifier, X, y=y, cv=cv, scoring='accuracy')

            self.assertEqual(len(scores), n_iterations)

    def assertClassifierPredictsProbabilities(self, classifier, sparsity_indicator='sparse'):
        for X, y in self.get_multilabel_data_for_tests(sparsity_indicator):
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
            classifier.fit(X_train, y_train)
            result = classifier.predict_proba(X_test)
            result = result.todense()

            max_value = result.max()
            min_value = result.min()

            self.assertEqual(result.shape, (X_test.shape[0], y.shape[1]))
            self.assertGreaterEqual(np.round(max_value), 0.0)
            self.assertGreaterEqual(np.round(min_value), 0.0)
            self.assertLessEqual(np.round(min_value), 1.0)
            self.assertLessEqual(np.round(max_value), 1.0)


if __name__ == '__main__':
    unittest.main()
