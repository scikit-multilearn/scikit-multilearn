import numpy as np
from sklearn.neighbors import NearestNeighbors
from ._mdsw import MDSW
from copy import copy

class CLEMS:
    def __init__(self, cost, dimension, params={}):
        self.cost = cost
        self.dimension = dimension

        if self.cost == 'ham':
            self.dis = hamming_loss
        elif self.cost == 'rank':
            self.dis = rank_loss
        elif self.cost == 'f1':
            self.dis = lambda x1, x2: 1.0 - f1_score(x1, x2)
        elif self.cost == 'acc':
            self.dis = lambda x1, x2: 1.0 - accuracy_score(x1, x2)

    def fit_transform(self, x_train, y_train):
        self.n_labels_ = y_train.shape[1]

        # get unique label combinations
        if sp.issparse(y_train):
            _, idx = np.unique(y_train.rows, return_index=True)
        else:
            _, idx = np.unique(y_train, return_index=True)

        y_unique = y_train[:,idx]

        self.knn_ = NearestNeighbors(n_neighbors=1)
        self.knn_.fit(y_unique)

        nearest_points = self.knn_.kneighbors(y_train)[1][:,0]
        nearest_points_counts = np.unique(nearest_points, return_counts=True)[1]

        # calculate delta matrix
        delta = np.zeros((2 * y_unique.shape[0], 2 * y_unique.shape[0]))
        for i in range(y_unique.shape[0]):
            for j in range(y_unique.shape[0]):
                delta[i, y_unique.shape[0] + j] = np.sqrt(self.dis(y_unique[None, i], y_unique[None, j]))
                delta[y_unique.shape[0] + j, i] = delta[i, y_unique.shape[0] + j]

        # calculate MDS embedding
        params = copy(self.params)
        params['n_components'] = self.dimension
        params['n_uq'] = y_unique.shape[0]
        params['uq_weight'] = nearest_points_counts
        mds = MDSW(**params)

        return mds.fit(delta).embedding_
