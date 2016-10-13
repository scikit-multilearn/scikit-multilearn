from ..base import MLClassifierBase

import numpy as np

##TODO: Check bias
class MLTSVM(MLClassifierBase):
    def __init__(self, c_k, sor_omega =1, lambda_param=1):
        super(MLClassifierBase, self).__init__()
        self.copyable_attrs = []
        self.lambda_param = lambda_param # TODO: possibility to set-up different lambda to different class
        self.c_k = c_k
        self.sor_omega = sor_omega
        # TODO: Add attributes after recognising with one is need

    def get_x_class_instanes(self, X, Y, label_class):
        return X[Y[label_class, :] != 0, :]  # TODO: hide warrnig, or replace by slower X[np.where(Y[label, :]) != 0, :]

    def get_x_noclass_instanes(self, X, Y, label_class):
        return X[Y[label_class, :] == 0, :]  # TODO: hide warrnig, or replace by slower X[np.where(Y[label, :]) == 0, :]

    def fit(self, X, Y):
        self.k = Y.shape[1]  # Count of classes
        m = X.shape[1]  # Count of features
        self.wk_bk = np.zeros([self.k, m + 1], dtype=float)
        for klasa in range(0, self.k):
            # Calculate Q for overrelaxation
            X_k = self.get_x_class_instanes(X, Y, klasa)
            X_nok = self.get_x_noclass_instanes(X, Y, klasa)
            e_k = np.ones((X_k.shape[0], 1), dtype=X.dtype)
            e_nok = np.ones((X_nok.shape[0], 1), dtype=X.dtype)
            H_k = np.concatenate((X_k, e_k), axis=1)  # horizontally
            G_k = np.concatenate((X_nok, e_nok), axis=1)  # horizontally
            Q_knoPrefixGk = np.linalg.inv(H_k.T * H_k + self.lambda_param * np.identity(m + 1)) * G_k.T
            Q_k = G_k * Q_knoPrefixGk
            Q_k = (Q_k + Q_k.T)/2   #It's step for succes requirments from quadratic problem

            # Calculate other
            alpha = self.successive_overrelaxation(self.sor_omega, Q_k)
            self.wk_bk[klasa] = (-Q_knoPrefixGk * alpha).T
        self.wk_norms = np.linalg.norm(self.wk_bk, axis=1)
        self.treshold = 1 / np.max(self.wk_norms)

    def successive_overrelaxation(self, omegaW, Q):
        # Initialization
        D = np.diag(Q)  # Only one dimension vector - is enough
        D_inv = 1.0 / D  # D-1 simplify form
        small_l = Q.shape[1]
        oldnew_alpha = np.zeros([small_l, 1])  # it's buffer
        threshold = 1e-6
        is_not_enough = True
        while is_not_enough:  # do while
            oldAlpha = oldnew_alpha
            for j in range(small_l - 1, -1):  # It's from last alpha to first
                oldnew_alpha[j] = oldAlpha[j] - omegaW * D_inv[j] * (Q[j, :] * oldnew_alpha - 1)
            oldnew_alpha = oldnew_alpha.clip(0, self.c_k)
            is_not_enough = np.linalg.norm(
                oldnew_alpha - oldAlpha) > threshold  # TODO: and any(oldnew_alpha != oldAlpha)
        return oldnew_alpha

    def predict(self, X):
        e = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_with1 = np.concatenate((X, e), axis=1)  # horizontaly
        wk_norms_multiplicated = self.wk_norms[np.newaxis, :]  # chage to form [[wk1, wk2, ..., wkk]]
        all_distances = (X_with1 * self.wk_bk.T) / wk_norms_multiplicated
        predicted_y = np.where(all_distances > self.treshold, 1, 0)
        # TODO: add label if no labels is in row
        return predicted_y
