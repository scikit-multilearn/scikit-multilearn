import numpy as np
from .base import MLClassifierBase
from scipy.sparse import csr_matrix


class MockClassifier(MLClassifierBase):
    """A stub classifier"""

    def __init__(self):
        super(MockClassifier, self).__init__()

    def fit(self, X, y):
        self.label_count = y.shape[1]
        return self

    def predict(self, X):
        return csr_matrix(np.ones(shape=(X.shape[0], self.label_count), dtype=int))
