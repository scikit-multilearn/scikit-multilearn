# Authors: Grzegorz Kulakowski <grzegorz7w@gmail.com>
# License: BSD 3 clause
from skmultilearn.base import MLClassifierBase

import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm
from scipy.sparse.linalg import inv as inv_sparse
from scipy.linalg import inv as inv_dense


class MLTSVM(MLClassifierBase):
    """Twin multi-Label Support Vector Machines

    Parameters
    ----------
    c_k : int
        the empirical risk penalty parameter that determines the trade-off between the loss terms
    sor_omega: float (default is 1.0)
        the smoothing parameter
    threshold : int (default is 1e-6)
            threshold above which a label should be assigned
    lambda_param : float (default is 1.0)
            the regularization parameter
    max_iteration : int (default is 500)
            maximum number of iterations to use in successive overrelaxation


    References
    ----------

    If you use this classifier please cite the original paper introducing the method:

    .. code :: bibtex

        @article{chen2016mltsvm,
          title={MLTSVM: a novel twin support vector machine to multi-label learning},
          author={Chen, Wei-Jie and Shao, Yuan-Hai and Li, Chun-Na and Deng, Nai-Yang},
          journal={Pattern Recognition},
          volume={52},
          pages={61--74},
          year={2016},
          publisher={Elsevier}
        }


    Examples
    --------

    Here's a very simple example of using MLTSVM with a fixed number of neighbors:

    .. code :: python

        from skmultilearn.adapt import MLTSVM

        classifier = MLTSVM(c_k = 2**-1)

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)


    You can also use :class:`~sklearn.model_selection.GridSearchCV` to find an optimal set of parameters:

    .. code :: python

        from skmultilearn.adapt import MLTSVM
        from sklearn.model_selection import GridSearchCV

        parameters = {'c_k': [2**i for i in range(-5, 5, 2)]}
        score = 'f1-macro

        clf = GridSearchCV(MLTSVM(), parameters, scoring=score)
        clf.fit(X, y)

        print (clf.best_params_, clf.best_score_)

        # output
        {'c_k': 0.03125} 0.347518217573


    """

    def __init__(self, c_k=0, sor_omega=1.0, threshold=1e-6, lambda_param=1.0, max_iteration=500):
        super(MLClassifierBase, self).__init__()
        self.max_iteration = max_iteration
        self.threshold = threshold
        self.lambda_param = lambda_param  # TODO: possibility to add different lambda to different labels
        self.c_k = c_k
        self.sor_omega = sor_omega
        self.copyable_attrs = ['c_k', 'sor_omega', 'lambda_param', 'threshold', 'max_iteration']

    def fit(self, X, Y):
        n_labels = Y.shape[1]
        m = X.shape[1]  # Count of features
        self.wk_bk = np.zeros([n_labels, m + 1], dtype=float)

        if sp.issparse(X):
            identity_matrix = sp.identity(m + 1)
            _inv = inv_sparse
        else:
            identity_matrix = np.identity(m + 1)
            _inv = inv_dense

        X_bias = _hstack(X, np.ones((X.shape[0], 1), dtype=X.dtype))
        self.iteration_count = []
        for label in range(0, n_labels):
            # Calculate the parameter Q for overrelaxation
            H_k = _get_x_class_instances(X_bias, Y, label)
            G_k = _get_x_noclass_instances(X_bias, Y, label)
            Q_knoPrefixGk = _inv((H_k.T).dot(H_k) + self.lambda_param * identity_matrix).dot(G_k.T)
            Q_k = G_k.dot(Q_knoPrefixGk).A
            Q_k = (Q_k + Q_k.T) / 2.0

            # Calculate other
            alpha_k = self._successive_overrelaxation(self.sor_omega, Q_k)
            if sp.issparse(X):
                self.wk_bk[label] = -Q_knoPrefixGk.dot(alpha_k).T
            else:
                self.wk_bk[label] = (-np.dot(Q_knoPrefixGk, alpha_k)).T

        self.wk_norms = norm(self.wk_bk, axis=1)
        self.treshold = 1.0 / np.max(self.wk_norms)

    def predict(self, X):
        X_with_bias = _hstack(X, np.ones((X.shape[0], 1), dtype=X.dtype))
        wk_norms_multiplicated = self.wk_norms[np.newaxis, :]  # change to form [[wk1, wk2, ..., wkk]]
        all_distances = (-X_with_bias.dot(self.wk_bk.T)) / wk_norms_multiplicated
        predicted_y = np.where(all_distances < self.treshold, 1, 0)
        # TODO: It's possible to add condition to: add label if no labels is in row.
        return predicted_y

    def _successive_overrelaxation(self, omegaW, Q):
        # Initialization
        D = np.diag(Q)  # Only one dimension vector - is enough
        D_inv = 1.0 / D  # D-1 simplify form
        small_l = Q.shape[1]
        oldnew_alpha = np.zeros([small_l, 1])  # buffer

        is_not_enough = True
        was_going_down = False
        last_alfa_norm_change = -1

        nr_iter = 0
        while is_not_enough:  # do while
            oldAlpha = oldnew_alpha
            for j in range(0, small_l):  # It's from last alpha to first
                oldnew_alpha[j] = oldAlpha[j] - omegaW * D_inv[j] * (Q[j, :].T.dot(oldnew_alpha) - 1)
            oldnew_alpha = oldnew_alpha.clip(0.0, self.c_k)
            alfa_norm_change = norm(oldnew_alpha - oldAlpha)

            if not was_going_down and last_alfa_norm_change > alfa_norm_change:
                was_going_down = True
            is_not_enough = alfa_norm_change > self.threshold and \
                            nr_iter < self.max_iteration \
                            and ((not was_going_down) or last_alfa_norm_change > alfa_norm_change)
            # TODO: maybe add any(oldnew_alpha != oldAlpha)

            last_alfa_norm_change = alfa_norm_change
            nr_iter += 1
        self.iteration_count.append(nr_iter)
        return oldnew_alpha


def _get_x_noclass_instances(X, Y, label_class):
    if sp.issparse(Y):
        indices = np.where(Y[:, 1].A == 0)[0]
    else:
        indices = np.where(Y[:, 1] == 0)[0]
    return X[indices, :]


def _get_x_class_instances(X, Y, label_class):
    if sp.issparse(Y):
        indices = Y[:, label_class].nonzero()[0]
    else:
        indices = np.nonzero(Y[:, label_class])[0]
    return X[indices, :]


def _hstack(X, Y):
    if sp.issparse(X):
        return sp.hstack([X, Y], format=X.format)
    else:
        return np.hstack([X, Y])
