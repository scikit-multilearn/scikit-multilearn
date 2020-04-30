from ..base.problem_transformation import ProblemTransformationBase

from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
import skmultilearn.tools as tools
from ..base.base import MLClassifierBase
from scipy.sparse import issparse, hstack

import copy, math
import numpy as np


class PTGridSearchCV(ProblemTransformationBase):
    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):

        super(PTGridSearchCV, self).__init__()
        self.classifiers_ = []
        self.estimator = estimator

        self.estimator.classifier = param_grid['classifier']
        del param_grid['classifier']

        if scoring == 'cll_loss' :
            scoring = make_scorer(tools.log_likelihood_loss, greater_is_better=False, needs_proba=True)

        if type(self.estimator).__name__ == 'InstanceBasedLogisticRegression':
            self.param_grid = param_grid
            param_grid = {'n_neighbors': [30]}
            self.gsc = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid,
                                    scoring=scoring, n_jobs=n_jobs, iid=iid,
                                    refit=refit, cv=cv, verbose=verbose,
                                    pre_dispatch=pre_dispatch, error_score=error_score,
                                    return_train_score=return_train_score)

        else :
            self.gsc = GridSearchCV(estimator=estimator.classifier, param_grid = param_grid,
                                    scoring=scoring, n_jobs=n_jobs, iid=iid,
                                    refit=refit, cv=cv, verbose=verbose,
                                    pre_dispatch=pre_dispatch, error_score=error_score,
                                    return_train_score=return_train_score)


    def fit(self, X, y):
        self.estimator.fit(X, y)

        if type(self.estimator).__name__ == 'ClassificationHeterogeneousFeature' :
            self.estimator.first_layer_ = self.find_optm_classifiers(X, y)

            class_membership = self.estimator.get_class_membership(self.estimator.first_layer_, X)
            X_concat_clm = self.estimator.concatenate_clm(X, class_membership)
            self.classifiers_ = self.find_optm_classifiers(X_concat_clm, y)

        elif type(self.estimator).__name__ == 'InstanceBasedLogisticRegression' :
            self.estimator.knn_layer = self.find_optm_classifiers(X, y)

            class_membership = self.estimator.get_class_membership(self.estimator.knn_layer, X)
            X_concat_clm = self.estimator.concatenate_class_membership(X, class_membership)
            self.gsc = GridSearchCV(estimator=self.estimator.classifier, param_grid = self.param_grid)
            self.classifiers_ = self.find_optm_classifiers(X_concat_clm, y)

        else :
            self.classifiers_ = self.find_optm_classifiers(X, y)

        return self


    def predict(self, X):
        self.estimator.classifiers_ = self.classifiers_
        return self.estimator.predict(self._ensure_input_format(X))

    def predict_proba(self, X):
        self.estimator.classifiers_ = self.classifiers_
        return self.estimator.predict_proba(self._ensure_input_format(X))

    def find_optm_classifiers(self, X, y):
        optimized_clfs_ = []
        if type(self.estimator).__name__ == 'ClassifierChain':
            optimized_clfs_ = [None for x in range(y.shape[1])]
            for i in self.estimator._order():
                gridsearchCV_ = copy.deepcopy(self.gsc)
                y_subset = self._generate_data_subset(y, i, axis=1)
                gridsearchCV_.fit(self._ensure_input_format(
                    X), self._ensure_output_format(y_subset))
                optimized_clfs_[i] = gridsearchCV_.best_estimator_
                X = hstack([X, y_subset])
        else :
            for i in range(len(self.estimator.classifiers_)) :
                gridsearchCV_ = copy.deepcopy(self.gsc)
                y_subset = self._generate_data_subset(y, i, axis=1)
                if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                    y_subset = np.ravel(y_subset.toarray())
                gridsearchCV_.fit(self._ensure_input_format(
                    X), self._ensure_output_format(y_subset))
                optimized_clf = gridsearchCV_.best_estimator_
                optimized_clfs_.append(optimized_clf)
        return optimized_clfs_
