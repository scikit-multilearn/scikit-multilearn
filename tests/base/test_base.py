# -*- coding: utf-8 -*-

"""Test cases for skmultilearn.base module"""

# Import from builtins
from builtins import zip
from builtins import range

# Import modules
import itertools
import unittest
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator

# Import from package
from skmultilearn.base.problem_transformation import ProblemTransformationBase

# Declare sparse formats
SPARSE_MATRIX_FORMATS = ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]

class ProblemTransformationBaseTest(unittest.TestCase):
    """Test cases for skmultilearn.base.problem_transformation"""

    def dense_and_dense_matrices_are_the_same(self, X, ensured_X):
        """Helper function to check similarity between dense matrices"""
        self.assertEqual(len(X), len(ensured_X))
        for row in range(len(X)):
            self.assertEqual(len(X[row]), len(ensured_X[row]))
            for col in range(len(X[row])):
                self.assertEqual(X[row][col], ensured_X[row][col])

    def dense_and_sparse_matrices_are_the_same(self, X, ensured_X):
        """Helper function to check similarity between dense and sparse matrices"""
        self.assertEqual(len(X), ensured_X.shape[0])
        for row in range(len(X)):
            self.assertEqual(len(X[row]), ensured_X.shape[1])
            for col in range(len(X[row])):
                self.assertEqual(X[row][col], ensured_X[row, col])

    def sparse_and_sparse_matrices_are_the_same(self, X, ensured_X):
        """Helper function to check similarity between sparse matrices"""
        self.assertEqual(X.shape, ensured_X.shape)
        # compare sparse matrices per
        # http://stackoverflow.com/questions/30685024/check-if-two-scipy-sparse-csr-matrix-are-equal
        self.assertTrue((X != ensured_X).nnz == 0)

    def test_if_require_dense_is_correctly_set(self):
        values = [True, False, [True, False], [
            True, True], [False, False], [False, True]]
        expected_values = [[True, True], [False, False], [
            True, False], [True, True], [False, False], [False, True]]

        for value, expected_value in zip(values, expected_values):
            classifier = ProblemTransformationBase(
                classifier=None, require_dense=value)
            self.assertEqual(classifier.require_dense, expected_value)

    def test_if_require_dense_is_correctly_inferred_when_none_passed(self):
        values = [None, ProblemTransformationBase()]
        expected_values = [[True, True], [False, False]]

        for value, expected_value in zip(values, expected_values):
            classifier = ProblemTransformationBase(
                classifier=value, require_dense=None)
            self.assertEqual(classifier.require_dense, expected_value)

    def test_make_sure_abstract_methods_are_not_implemented_in_base(self):
        classifier = ProblemTransformationBase()

        with self.assertRaises(NotImplementedError):
            classifier.fit([], [])

        with self.assertRaises(NotImplementedError):
            classifier.predict([])

    def test_make_sure_params_include_all_params(self):
        classifier = ProblemTransformationBase()
        classifier_params = classifier.get_params(deep=False)
        expected_params = ['classifier', 'require_dense']

        for param in expected_params:
            self.assertIn(param, classifier_params)

    def test_make_sure_ml_base_classifier_follows_base_estimator(self):
        classifier = ProblemTransformationBase()
        self.assertIsInstance(classifier, BaseEstimator)

    def test_ensure_input_format_returns_dense_from_dense_if_required(self):
        classifier = ProblemTransformationBase(require_dense=True)

        X = np.zeros((2, 3))
        ensured_X = classifier.ensure_input_format(X)

        self.assertTrue(not sp.issparse(ensured_X))
        self.dense_and_dense_matrices_are_the_same(X, ensured_X)

    def test_ensure_input_format_returns_dense_from_sparse_if_required(self):
        classifier = ProblemTransformationBase(require_dense=True)

        X = sp.csr_matrix(np.zeros((2, 3)))
        ensured_X = classifier.ensure_input_format(X)

        self.assertTrue(not sp.issparse(ensured_X))
        self.dense_and_sparse_matrices_are_the_same(ensured_X, X)

    def test_ensure_input_format_returns_sparse_from_dense_if_required(self):
        classifier = ProblemTransformationBase(require_dense=False)

        X = np.zeros((2, 3))
        ensured_X = classifier.ensure_input_format(X)

        self.assertTrue(sp.issparse(ensured_X))
        self.dense_and_sparse_matrices_are_the_same(X, ensured_X)

    def test_ensure_input_format_returns_sparse_from_sparse_if_required(self):
        classifier = ProblemTransformationBase(require_dense=False)

        X = sp.csr_matrix(np.zeros((2, 3)))
        ensured_X = classifier.ensure_input_format(X)

        self.assertTrue(sp.issparse(ensured_X))
        self.sparse_and_sparse_matrices_are_the_same(X, ensured_X)

    def test_ensure_input_format_returns_sparse_from_dense_if_enforced(self):
        for require_dense in itertools.product([True, False], repeat=2):
            classifier = ProblemTransformationBase(require_dense=require_dense)

            X = np.zeros((2, 3))
            ensured_X = classifier.ensure_input_format(X, enforce_sparse=True)

            self.assertTrue(sp.issparse(ensured_X))
            self.dense_and_sparse_matrices_are_the_same(X, ensured_X)

    def test_ensure_input_format_returns_sparse_from_sparse_if_enforced(self):
        for require_dense in itertools.product([True, False], repeat=2):
            classifier = ProblemTransformationBase(require_dense=require_dense)

            X = sp.csr_matrix(np.zeros((2, 3)))
            ensured_X = classifier.ensure_input_format(X, enforce_sparse=True)

            self.assertTrue(sp.issparse(ensured_X))
            self.sparse_and_sparse_matrices_are_the_same(X, ensured_X)

    def test_ensure_input_format_returns_sparse_in_format_from_dense_if_enforced(self):
        for sparse_format in SPARSE_MATRIX_FORMATS:
            for require_dense in [True, False]:
                classifier = ProblemTransformationBase(
                    require_dense=require_dense)

                X = np.zeros((2, 3))
                ensured_X = classifier.ensure_input_format(
                    X, sparse_format=sparse_format, enforce_sparse=True)

                self.assertTrue(sp.issparse(ensured_X))
                self.assertEqual(ensured_X.format, sparse_format)

    def test_ensure_input_format_returns_sparse_in_format_from_dense_if_required(self):
        for sparse_format in SPARSE_MATRIX_FORMATS:
            classifier = ProblemTransformationBase(require_dense=False)

            X = np.zeros((2, 3))
            ensured_X = classifier.ensure_input_format(
                X, sparse_format=sparse_format)

            self.assertTrue(sp.issparse(ensured_X))
            self.assertEqual(ensured_X.format, sparse_format)

    def test_ensure_output_format_returns_dense_from_dense_if_required(self):
        classifier = ProblemTransformationBase(require_dense=True)

        y = np.zeros((2, 3))
        ensured_y = classifier.ensure_output_format(y)

        self.assertTrue(not sp.issparse(ensured_y))
        self.dense_and_dense_matrices_are_the_same(y, ensured_y)

    def test_ensure_output_format_returns_dense_from_sparse_if_required(self):
        classifier = ProblemTransformationBase(require_dense=True)

        y = sp.csr_matrix(np.zeros((2, 3)))
        ensured_y = classifier.ensure_output_format(y)

        self.assertTrue(not sp.issparse(ensured_y))
        self.dense_and_sparse_matrices_are_the_same(ensured_y, y)

    def test_ensure_output_format_returns_sparse_from_dense_if_required(self):
        classifier = ProblemTransformationBase(require_dense=False)

        y = np.zeros((2, 3))
        ensured_y = classifier.ensure_output_format(y)

        self.assertTrue(sp.issparse(ensured_y))
        self.dense_and_sparse_matrices_are_the_same(y, ensured_y)

    def test_ensure_output_format_returns_sparse_from_sparse_if_required(self):
        classifier = ProblemTransformationBase(require_dense=False)

        y = sp.csr_matrix(np.zeros((2, 3)))
        ensured_y = classifier.ensure_output_format(y)

        self.assertTrue(sp.issparse(ensured_y))
        self.sparse_and_sparse_matrices_are_the_same(y, ensured_y)

    def test_ensure_output_format_returns_sparse_from_dense_if_enforced(self):
        for require_dense in itertools.product([True, False], repeat=2):
            classifier = ProblemTransformationBase(require_dense=require_dense)

            y = np.zeros((2, 3))
            ensured_y = classifier.ensure_output_format(y, enforce_sparse=True)

            self.assertTrue(sp.issparse(ensured_y))
            self.dense_and_sparse_matrices_are_the_same(y, ensured_y)

    def test_ensure_output_format_returns_sparse_from_sparse_if_enforced(self):
        for require_dense in itertools.product([True, False], repeat=2):
            classifier = ProblemTransformationBase(require_dense=require_dense)

            y = sp.csr_matrix(np.zeros((2, 3)))
            ensured_y = classifier.ensure_output_format(y, enforce_sparse=True)

            self.assertTrue(sp.issparse(ensured_y))
            self.sparse_and_sparse_matrices_are_the_same(y, ensured_y)

    def test_ensure_output_format_returns_sparse_in_format_from_dense_if_enforced(self):
        for sparse_format in SPARSE_MATRIX_FORMATS:
            for require_dense in [True, False]:
                classifier = ProblemTransformationBase(
                    require_dense=require_dense)

                y = np.zeros((2, 3))
                ensured_y = classifier.ensure_output_format(
                    y, sparse_format=sparse_format, enforce_sparse=True)

                self.assertTrue(sp.issparse(ensured_y))
                self.assertEqual(ensured_y.format, sparse_format)

    def test_ensure_output_format_returns_sparse_in_format_from_dense_if_required(self):
        for sparse_format in SPARSE_MATRIX_FORMATS:
            classifier = ProblemTransformationBase(require_dense=False)

            y = np.zeros((2, 3))
            ensured_y = classifier.ensure_output_format(
                y, sparse_format=sparse_format)

            self.assertTrue(sp.issparse(ensured_y))
            self.assertEqual(ensured_y.format, sparse_format)

    def test_ensure_output_is_1d_for_single_label_y_when_dense(self):
        for input_sparse in [True, False]:
            classifier = ProblemTransformationBase(require_dense=True)
            y_single = np.ones((2, 1), dtype=int)

            shape = y_single.shape
            if input_sparse:
                y_single = sp.csr_matrix(y_single)

            ensured_y_single = classifier.ensure_output_format(y_single)

            self.assertIsInstance(ensured_y_single, np.ndarray)
            self.assertEqual(len(ensured_y_single.shape), 1)
            self.assertEqual(ensured_y_single.shape[0], shape[0])

if __name__ == '__main__':
    unittest.main()
