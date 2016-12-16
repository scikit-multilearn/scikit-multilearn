# Authors: Grzegorz Kulakowski <grzegorz7w@gmail.com>
# License: BSD 3 clause
from ..base import MLClassifierBase

import numpy as np
from scipy import sparse
from scipy.linalg import inv, norm


class MLTSVM(MLClassifierBase):
    def __init__(self, c_k, sor_omega =1.0, threshold = 1e-6, lambda_param=1.0, max_iteration=500):
        super(MLClassifierBase, self).__init__()
        self.max_sor_iteration = max_iteration
        self.threshold = threshold
        self.copyable_attrs = []
        self.lambda_param = lambda_param  # TODO: possibility to add different lambda to different labels
        self.c_k = c_k
        self.sor_omega = sor_omega
        self.copyable_attrs = ['c_k','sor_omega','lambda_param', 'threshold', 'max_iteration']

    def get_x_class_instanes(self, X, Y, label_class):
        y_labels = Y[:, label_class] != 0
        return X[y_labels, :]  # TODO: hide warning, or replace by slower X[np.where(Y[label, :]) != 0, :]

    def get_x_noclass_instanes(self, X, Y, label_class):
        return X[Y[:,label_class] == 0, :]  # TODO: hide warning, or replace by slower X[np.where(Y[label, :]) == 0, :]

    def fit(self, X, Y):
        self.k = Y.shape[1]  # Count of classes
        m = X.shape[1]  # Count of features
        self.wk_bk = np.zeros([self.k, m + 1], dtype=float)
        X_bias = np.concatenate((X, np.ones((X.shape[0], 1), dtype=X.dtype)), axis=1)
        self.iteration_count = []
        for label in range(0, self.k):
            # Calculate the parameter Q for overrelaxation
            H_k = self.get_x_class_instanes(X_bias, Y, label)
            G_k = self.get_x_noclass_instanes(X_bias, Y, label)
            Q_knoPrefixGk = inv((H_k.T).dot(H_k) + self.lambda_param * np.identity(m + 1)).dot(G_k.T)
            Q_k = G_k.dot(Q_knoPrefixGk)
            # It's step for success requirements from quadratic problem - symetric matrix
            Q_k = (Q_k + Q_k.T)/2.0

            # Calculate other
            alpha_k = self.__successive_overrelaxation(self.sor_omega, Q_k)
            self.wk_bk[label] = (-np.dot(Q_knoPrefixGk,alpha_k)).T

            # Garbage Collect
            H_k = None
            G_k = None
            Q_knoPrefixGk = None
            Q_k = None
        self.wk_norms = norm(self.wk_bk, axis=1)
        self.treshold = 1.0 / np.max(self.wk_norms)

    def __successive_overrelaxation(self, omegaW, Q):
        # Initialization
        D = np.diag(Q)  # Only one dimension vector - is enough
        D_inv = 1.0 / D  # D-1 simplify form
        small_l = Q.shape[1]
        oldnew_alpha = np.zeros([small_l, 1])  # it's buffer

        is_not_enough = True
        was_going_down = False
        last_alfa_norm_change = -1

        nr_iter = 0
        while is_not_enough:  # do while
            oldAlpha = oldnew_alpha
            for j in range(0,small_l):  # It's from last alpha to first
                oldnew_alpha[j] = oldAlpha[j] - omegaW * D_inv[j] * (Q[j, :].T.dot(oldnew_alpha) - 1)
            oldnew_alpha = oldnew_alpha.clip(0.0, self.c_k)
            alfa_norm_change = norm(oldnew_alpha - oldAlpha)

            if not was_going_down and last_alfa_norm_change > alfa_norm_change:
                was_going_down = True
            is_not_enough = alfa_norm_change > self.threshold and\
                            nr_iter < self.max_sor_iteration \
                            and ((not was_going_down) or last_alfa_norm_change > alfa_norm_change)
                            # TODO: maybe add any(oldnew_alpha != oldAlpha)

            last_alfa_norm_change = alfa_norm_change
            nr_iter+=1
        self.iteration_count.append(nr_iter)
        return oldnew_alpha

    def predict(self, X):
        e = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_with_bias = np.concatenate((X, e), axis=1)  # horizontaly
        wk_norms_multiplicated = self.wk_norms[np.newaxis, :]  # change to form [[wk1, wk2, ..., wkk]]
        all_distances = (-X_with_bias.dot(self.wk_bk.T)) / wk_norms_multiplicated
        predicted_y = np.where(all_distances < self.treshold, 1, 0)
        # TODO: It's possible to add condition to: add label if no labels is in row.
        return predicted_y
