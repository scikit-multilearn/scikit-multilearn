import pytest
import numpy as np
import unittest
from sklearn.datasets import load_iris, make_multilabel_classification
import sys
from skmultilearn.missing.smile import SMiLE
from skmultilearn.missing.functions import *


class SmileTest(unittest.TestCase):
    def test_weight(self):
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        W = weight_adjacent_matrix(X=X, k=5)

        self.assertTrue(np.sum(W) != 0)
        self.assertEqual(W.shape[0], X.shape[1])
        self.assertEqual(W.shape[1], X.shape[1])

    def test_label_correlation(self):
    
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        correlation = np.zeros(shape=[y.shape[0], y.shape[0]])
        correlation = label_correlation(y, s=0.5)

        self.assertTrue(np.sum(correlation) != 0)
        self.assertEqual(correlation.shape[0], y.shape[0])
        self.assertEqual(correlation.shape[1], y.shape[0])

    
    def test_estimate_missing_labels(self):

        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        correlation = label_correlation(y, s=0.5)
        estimate_matrix = np.zeros(shape=[y.shape[0], y.shape[1]])
        estimate_matrix = estimate_mising_labels(y, correlation)

        self.assertTrue(np.sum(estimate_matrix) != 0)
        self.assertEqual(estimate_matrix.shape[0], y.shape[0])
        self.assertEqual(estimate_matrix.shape[1], y.shape[1])
    
    def test_diagonal_matrix_H(self):
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        H = np.zeros(shape=[X.shape[1], X.shape[1]])
        H = diagonal_matrix_H(X, y)

        self.assertTrue(np.sum(H) != 0)
        self.assertEqual(H.shape[0], X.shape[1])
        self.assertEqual(H.shape[1], X.shape[1])
    
    def test_diagonal_matrix_lambda(self):
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        W = weight_adjacent_matrix(X=X, k=5)
        diagonal_lambda = np.zeros(shape=[W.shape[0], W.shape[1]])
        diagonal_lambda = diagonal_matrix_lambda(W)

        self.assertTrue(np.sum(diagonal_lambda) != 0)
        self.assertEqual(diagonal_lambda.shape[0], W.shape[0])
        self.assertEqual(diagonal_lambda.shape[1], W.shape[1])

    def test_laplacian_matrix(self):
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        W = weight_adjacent_matrix(X=X, k= 5)
        diagonal_lambda = diagonal_matrix_lambda(W)
        M = np.zeros(shape=[X.shape[0], X.shape[0]])
        M = graph_laplacian_matrix(diagonal_lambda, W)
        self.assertTrue(np.sum(np.diag(M)) != 0)
        self.assertEqual(M.shape[0], W.shape[0])
        self.assertEqual(M.shape[1], W.shape[1])

    def test_diagonal_matrix_Hc(self):
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        H = diagonal_matrix_H(X, y)
        Hc = np.zeros(shape = [H.shape[0], H.shape[1]])
        Hc = diagonal_matrix_Hc(H)
        self.assertTrue(np.sum(Hc) != 0)

    def test_predictive_matrix(self):
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        L = label_correlation(y, s=0.5)
        estimate_matrix = estimate_mising_labels(y, L)
        H = diagonal_matrix_H(X, y)
        Hc = diagonal_matrix_Hc(H)
        W = weight_adjacent_matrix(X,k=5)
        lambda_matrix = diagonal_matrix_lambda(W)
        M = graph_laplacian_matrix(lambda_matrix, W)
        P = np.zeros(shape= [X.shape[1], y.shape[1]])
        P = predictive_matrix(X, Hc, M, estimate_matrix, alpha=0.35)
        self.assertTrue(np.sum(P) != 0)

    def test_label_bias(self):
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        L = label_correlation(y, s=0.5)
        estimate_matrix = estimate_mising_labels(y, L)
        H = diagonal_matrix_H(X, y)
        Hc = diagonal_matrix_Hc(H)
        W = weight_adjacent_matrix(X,k=5)
        lambda_matrix = diagonal_matrix_lambda(W)
        M = graph_laplacian_matrix(lambda_matrix, W)
        P = predictive_matrix(X, Hc, M, estimate_matrix, alpha=0.35)
        b = np.zeros(y.shape[1])
        b = label_bias(estimate_matrix, P, X, H)
        self.assertTrue(np.sum(b) != 0)

    
    def test_fit(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        smile.fit(X, y)
        self.assertTrue(np.sum(smile.P) != 0)
        self.assertTrue(np.sum(smile.b) != 0)


    def test_predict(self):
        smile = SMiLE()
        X, y = make_multilabel_classification()
        X = np.transpose(X)
        y = np.transpose(y)
        smile.fit(X,y)
        predictions = np.zeros(shape=[X.shape[0], y.shape[1]])
        predictions1, predictions2 = smile.predict(X)
        self.assertTrue(np.sum(predictions1) != 0)
        self.assertTrue(np.sum(predictions2) != 0)

    def test_getParams(self):
        smile = SMiLE()
        s, alpha, k = smile.getParams()
        self.assertEqual(s, 0.5)
        self.assertEqual(alpha, 0.35)
        self.assertEqual(k, 5)

    def test_setParams(self):
        smile = SMiLE()
        smile.setParams(0.85, 0.17, 7)
        s, alpha, k = smile.getParams()
        self.assertEqual(s, 0.85)
        self.assertEqual(alpha, 0.17)
        self.assertEqual(k, 7)

if __name__ == '__main__':
    unittest.main()
