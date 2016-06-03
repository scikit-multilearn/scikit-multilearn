import copy
import numpy as np
from ..utils import get_matrix_in_format, matrix_creation_function_for_format
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin


class MLClassifierBase(BaseEstimator, ClassifierMixin):
    """Base class providing API and common functions for all multi-label classifiers.

    Parameters
    ----------

    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.
    require_dense : boolean
        Whether the base classifier requires input as dense arrays, False by default
    """

    def __init__(self):
        super(MLClassifierBase, self).__init__()

        self.copyable_attrs = []

    def generate_data_subset(self, y, subset, axis):
        """This function subsets the array of binary label vectors to include only certain labels. 

        Parameters
        ----------

        y : array-like of array-likes
            An array-like of binary label vectors.

        subset: array-like of integers
            array of integers, indices that will be subsetted from array-likes in y

        axis: integer 0 for 'rows', 1 for 'labels', 
            control variable for whether to return rows or labels as indexed by subset

        Returns
        -------

        multi-label binary label vector : array-like of array-likes of {0,1}
            array of binary label vectors including label data only for labels from parameter labels
        """
        return_data = None
        if axis == 1:
            return_data = y.tocsc()[:, subset]
        elif axis == 0:
            return_data = y.tocsr()[subset, :]

        return return_data

    def ensure_input_format(self, X, sparse_format='csr', enforce_sparse=False):
        """This function ensures that input format follows the density/sparsity requirements of base classifier. 

        Parameters
        ----------

        X : array-like or sparse matrix, shape = [n_samples, n_features]
            An input feature matrix

        sparse_format: string
            Requested format of returned scipy.sparse matrix, if sparse is returned

        enforce_sparse : bool
            Ignore require_dense and enforce sparsity, useful internally

        Returns
        -------

        transformed X : array-like or sparse matrix, shape = [n_samples, n_features]
            If require_dense was set to true for input features in the constructor, 
            the returned value is an array-like of array-likes. If require_dense is 
            set to false, a sparse matrix of format sparse_format is returned, if 
            possible - without cloning.
        """
        is_sparse = issparse(X)

        if is_sparse:
            if self.require_dense[0] and not enforce_sparse:
                return X.toarray()
            else:
                if sparse_format is None:
                    return X
                else:
                    return get_matrix_in_format(X, sparse_format)
        else:
            if self.require_dense[0] and not enforce_sparse:
                # TODO: perhaps a check_array?
                return X
            else:
                return matrix_creation_function_for_format(sparse_format)(X)

    def ensure_output_format(self, y, sparse_format='csr', enforce_sparse=False):
        """This function ensures that output format follows the density/sparsity requirements of base classifier. 

        Parameters
        ----------

        y : array-like with shape = [n_samples] or [n_samples, n_outputs]; or sparse matrix, shape = [n_samples, n_outputs]  
            An input feature matrix

        sparse_format: string
            Requested format of returned scipy.sparse matrix, if sparse is returned

        enforce_sparse : bool
            Ignore require_dense and enforce sparsity, useful internally

        Returns
        -------

        transformed y: array-like with shape = [n_samples] or [n_samples, n_outputs]; or sparse matrix, shape = [n_samples, n_outputs]  
            If require_dense was set to True for input features in the constructor, 
            the returned value is an array-like of array-likes. If require_dense is 
            set to False, a sparse matrix of format sparse_format is returned, if 
            possible - without cloning.
        """
        is_sparse = issparse(y)

        if is_sparse:
            if self.require_dense[1] and not enforce_sparse:
                if y.shape[1] != 1:
                    return y.toarray()
                elif y.shape[1] == 1:
                    return np.ravel(y.toarray())
            else:
                if sparse_format is None:
                    return y
                else:
                    return get_matrix_in_format(y, sparse_format)
        else:
            if self.require_dense[1] and not enforce_sparse:
                # ensuring 1d
                if len(y[0]) == 1:
                    return np.ravel(y)
                else:
                    return y
            else:
                return matrix_creation_function_for_format(sparse_format)(y)

    def fit(self, X, y):
        """Abstract class to implement to fit classifier according to X,y.

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples, n_labels]
            Binary label vectors with 1 if label should be applied and 0 if not.

        Returns
        -------
        self : object
            Returns self.
        """
        raise NotImplementedError("MLClassifierBase::fit()")

    def predict(self, X):
        """Abstract class to implement to perform classification on an array of test vectors X.

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------

        y : array-like, shape = [n_samples, n_labels]
            Binary label vectors with 1 if label should be applied and 0 if not. n_labels is number of labels in the 
            multi-label instance that the classifier was fit to.

        """
        raise NotImplementedError("MLClassifierBase::predict()")

    def get_params(self, deep=True):
        """
        Introspection of classifier for search models like cross validation and grid
        search.
        Parameters
        ----------
        deep : boolean
            If true all params will be introspected also and appended to the output dict.
        Returns
        -------
        out : dictionary
            Dictionary of all parameters and their values. If deep=True the dictionary
            also holds the parameters of the parameters.
        """

        out = dict()

        for attr in self.copyable_attrs:
            out[attr] = getattr(self, attr)

            if hasattr(getattr(self, attr), 'get_params') and deep:
                deep_items = getattr(self, attr).get_params().items()
                out.update((attr + '__' + k, val) for k, val in deep_items)

        return out

    def set_params(self, **parameters):
        """
        Set parameters as returned by `get_params`.
        @see https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L243
        """

        if not parameters:
            return self

        valid_params = self.get_params(deep=True)

        for parameter, value in parameters.items():
            split = parameter.split('__', 1)

            if len(split) > 1:
                sub_obj_name, sub_param = split

                if sub_obj_name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))

                sub_object = valid_params[sub_obj_name]
                sub_object.set_params(**{sub_param: value})
            else:
                if parameter in valid_params:
                    setattr(self, parameter, value)
                else:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))

        return self
