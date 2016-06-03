from .base import MLClassifierBase


class MockClassifier(MLClassifierBase):
    """docstring for MockClassifier"""

    def __init__(self):
        super(MockClassifier, self).__init__()

    def fit(X, y):
        self.label_count = y.shape[1]
        return self

    def predict(X):
        return csr_matrix(np.ones(shape=(X.shape[0], self.label_count), dtype=int))
