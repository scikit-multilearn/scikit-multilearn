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


class StructuredGridSearchCV(ProblemTransformationBase):
    """Hyperparameter tuning per each label classifier

    As original GridSearchCV provided by scikit-learn ignores
    BR&CC structural property, it cannot search best parameter and classifier
    for each label. Therefore, StructuredGridSearchCV was implemented for fine tuning
    with considering structural property. StructuredGridSearchCV searches best classifier
    with optimal hyper-parameters for each labels.

    StructuredGridSearchCV provides "fit", "predict", "predict_proba" as its methods.
    It provides list of optimal classifiers with fine-tuned hyper-parameters
    via find_optm_classifier function.
    If print_best_param is True, find_optm_classifier function prints
    best parameter for each label.

    Parameters
    ----------
    Same as GridsearchCV in scikit-learn

    print_best_param : bool, default = False
        whether print best parameter each label classifier or not

    Attributes
    ----------
    classifiers_ : List[:class:`~sklearn.base.BaseEstimator`] of shape `n_labels`
        list of classifiers trained per partition, set in :meth:`fit`

    print_best_param : bool, default = False
        whether print best parameter each label classifier or not

    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    classifier_num : int
        The index which corresponds to each label classifier

    """
    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False, print_best_param = False):

        super(StructuredGridSearchCV, self).__init__()
        self.classifiers_ = []
        self.print_best_param = print_best_param
        self.estimator = estimator
        self.classifier_num = 0

        self.estimator.classifier = param_grid['classifier']
        del param_grid['classifier']


        if type(self.estimator).__name__ == 'InstanceBasedLogisticRegression':
            self.param_grid = param_grid
            param_grid = {'n_neighbors': [10]}
            self.gsc = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid,
                                    scoring=scoring, n_jobs=n_jobs, iid=iid,
                                    refit=refit, cv=cv, verbose=verbose,
                                    pre_dispatch=pre_dispatch, error_score=error_score,
                                    return_train_score=return_train_score)

        else:
            self.gsc = GridSearchCV(estimator=estimator.classifier, param_grid=param_grid,
                                    scoring=scoring, n_jobs=n_jobs, iid=iid,
                                    refit=refit, cv=cv, verbose=verbose,
                                    pre_dispatch=pre_dispatch, error_score=error_score,
                                    return_train_score=return_train_score)

    def fit(self, X, y):
        """Fits classifier to training data and finds optimal classifier

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        self
            fitted instance of self

        Notes
        -----
        .. note :: Input matrices are converted to sparse format internally if a numpy representation is passed
        """

        self.estimator.fit(X, y)

        if type(self.estimator).__name__ == 'ClassificationHeterogeneousFeature':
            self.estimator.first_layer_ = self.find_optm_classifiers(X, y)

            class_membership = self.estimator.get_class_membership(self.estimator.first_layer_, X)
            X_concat_clm = self.estimator.concatenate_clm(X, class_membership)
            self.classifiers_ = self.find_optm_classifiers(X_concat_clm, y)

        elif type(self.estimator).__name__ == 'InstanceBasedLogisticRegression':
            self.estimator.knn_layer = self.find_optm_classifiers(X, y)

            class_membership = self.estimator.get_class_membership(self.estimator.knn_layer, X)
            X_concat_clm = self.estimator.concatenate_class_membership(X, class_membership)
            self.gsc = GridSearchCV(estimator=self.estimator.classifier, param_grid=self.param_grid)
            self.classifiers_ = self.find_optm_classifiers(X_concat_clm, y)

        else:
            self.classifiers_ = self.find_optm_classifiers(X, y)

        return self

    def predict(self, X):
        """Predict labels for X using optimal classifiers

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """

        self.estimator.classifiers_ = self.classifiers_
        return self.estimator.predict(self._ensure_input_format(X))

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X using optimal classifiers

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `float in [0.0, 1.0]`, shape=(n_samples, n_labels)
            matrix with label assignment probabilities
        """

        self.estimator.classifiers_ = self.classifiers_
        return self.estimator.predict_proba(self._ensure_input_format(X))

    def find_optm_classifiers(self, X, y):
        """Find optimal classifier per lable

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        optimized_clfs_
            list of optimal classifier per label

        Notes
        -----
        .. note :: Input matrices are converted to sparse format internally if a numpy representation is passed
        """

        optimized_clfs_ = []
        if type(self.estimator).__name__ == 'ClassifierChain':
            optimized_clfs_ = [None for x in range(y.shape[1])]
            for i in self.estimator._order():
                gridsearchCV_ = copy.deepcopy(self.gsc)
                y_subset = self._generate_data_subset(y, i, axis=1)
                gridsearchCV_.fit(self._ensure_input_format(
                    X), self._ensure_output_format(y_subset))
                if self.print_best_param :
                    print("[", (self.classifier_num + 1), "]", " Classifier Best Parameters : ", gridsearchCV_.best_params_)
                optimized_clfs_[i] = gridsearchCV_.best_estimator_
                X = hstack([X, y_subset])
                self.classifier_num += 1
        else:
            for i in range(len(self.estimator.classifiers_)):
                gridsearchCV_ = copy.deepcopy(self.gsc)
                y_subset = self._generate_data_subset(y, i, axis=1)
                if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                    y_subset = np.ravel(y_subset.toarray())
                gridsearchCV_.fit(self._ensure_input_format(
                    X), self._ensure_output_format(y_subset))
                if self.print_best_param :
                    print("[", (self.classifier_num + 1), "]", " Classifier Best Parameters : ", gridsearchCV_.best_params_)
                optimized_clf = gridsearchCV_.best_estimator_
                optimized_clfs_.append(optimized_clf)
                self.classifier_num += 1
        return optimized_clfs_