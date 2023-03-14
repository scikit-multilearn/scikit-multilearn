from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator
from copy import copy


class Keras(BaseEstimator):
    def __init__(self, build_function, multi_class=False, keras_params=None):
        if not callable(build_function):
            raise ValueError("Model construction function must be callable.")

        self.multi_class = multi_class
        self.build_function = build_function
        if keras_params is None:
            keras_params = {}

        self.keras_params = keras_params

    def fit(self, X, y):
        if self.multi_class:
            self.n_classes_ = len(set(y))
        else:
            self.n_classes_ = 1

        build_callable = lambda: self.build_function(X.shape[1], self.n_classes_)
        keras_params = copy(self.keras_params)
        keras_params["build_fn"] = build_callable

        self.classifier_ = KerasClassifier(**keras_params)
        self.classifier_.fit(X, y)

    def predict(self, X):
        return self.classifier_.predict(X)
