from builtins import range
from .base import MLClassifierBase
import copy
import numpy as np


class RepeatClassifier(MLClassifierBase):
    """Simple classifier for handling cases where """

    def __init__(self):
        super(RepeatClassifier, self).__init__()

    def fit(self, X, y):
        self.value_to_repeat = copy.copy(y.tocsr[0, :])
        self.return_value = np.full(y)
        return self

    def predict(self, X):
        return np.array([np.copy(self.value_to_repeat) for x in range(len(X))])
