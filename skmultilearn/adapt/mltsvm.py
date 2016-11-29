# Authors: Grzegorz Kulakowski <grzegorz7w@gmail.com>
# License: BSD 3 clause

from ..base import MLClassifierBase

import numpy as np

##TODO: Check bias
class MLTSVM(MLClassifierBase):
    def __init__(self, c_k, sor_omega =1.0, threshold = 1e-6, lambda_param=1.0):
        super(MLClassifierBase, self).__init__()
        self.threshold = threshold
        self.copyable_attrs = []
        self.lambda_param = lambda_param # TODO: possibility to set-up different lambda to different class
        self.c_k = c_k
        self.sor_omega = sor_omega
        # TODO: Add attributes after recognising with one is need

    def get_x_class_instanes(self, X, Y, label_class):
        y_labels = Y[:,label_class] != 0
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
            # Calculate the parametr Q for overrelaxation
            H_k = self.get_x_class_instanes(X_bias, Y, label)
            G_k = self.get_x_noclass_instanes(X_bias, Y, label)
            Q_knoPrefixGk = np.dot(np.linalg.inv(np.dot(H_k.T, H_k) + self.lambda_param * np.identity(m + 1)), G_k.T)
            Q_k = np.dot(G_k,Q_knoPrefixGk)
            Q_k = (Q_k + Q_k.T)/2   #It's step for succes requirments from quadratic problem

            # Calculate other
            alpha_k = self.__successive_overrelaxation(self.sor_omega, Q_k)
            self.wk_bk[label] = (-np.dot(Q_knoPrefixGk,alpha_k)).T #TODO: check it
        self.wk_norms = np.linalg.norm(self.wk_bk, axis=1)
        self.treshold = 1.0 / np.max(self.wk_norms)

    def __successive_overrelaxation(self, omegaW, Q):
        # Initialization
        D = np.diag(Q)  # Only one dimension vector - is enough
        D_inv = 1.0 / D  # D-1 simplify form, TODO: check if is correct
        small_l = Q.shape[1]
        oldnew_alpha = np.zeros([small_l, 1])  # it's buffer

        is_not_enough = True
        was_going_down = False
        last_alfa_norm_change = -1

        nr_iter = 0
        max_nr_iter = 500
        while is_not_enough:  # do while
            oldAlpha = oldnew_alpha
            for j in range(small_l - 1, -1, -1):  # It's from last alpha to first
                oldnew_alpha[j] = oldAlpha[j] - omegaW * D_inv[j] * (np.dot(Q[j, :].T, oldnew_alpha) - 1)
            oldnew_alpha = oldnew_alpha.clip(0.0, self.c_k)
            alfa_norm_change = np.linalg.norm(oldnew_alpha - oldAlpha)
            #print alfa_norm_change

            if not was_going_down and last_alfa_norm_change > alfa_norm_change:
                was_going_down = True
            is_not_enough = alfa_norm_change > self.threshold and\
                            nr_iter < max_nr_iter\
                            and ((not was_going_down) or last_alfa_norm_change > alfa_norm_change) # TODO: and any(oldnew_alpha != oldAlpha)

            last_alfa_norm_change = alfa_norm_change
            nr_iter+=1
        self.iteration_count.append(nr_iter)
        return oldnew_alpha

    def predict(self, X):
        e = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_with_bias = np.concatenate((X, e), axis=1)  # horizontaly
        wk_norms_multiplicated = self.wk_norms[np.newaxis, :]  # chage to form [[wk1, wk2, ..., wkk]]
        all_distances = (-np.dot(X_with_bias,self.wk_bk.T)) / wk_norms_multiplicated
        predicted_y = np.where(all_distances < self.treshold, 1, 0)
        # TODO: add label if no labels is in row
        return predicted_y
