import numpy as np
from ..utils import get_matrix_in_format, matrix_creation_function_for_format
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin


class MLClassifierBase(BaseEstimator, ClassifierMixin):
    """Base class providing API and common functions for all multi-label
    classifiers.

    Implements base functionality for ML classifiers, especially the get/set params for
    scikit-learn compatibility.

    Attributes
    ----------
    copyable_attrs : List[str]
        list of attribute names that should be copied when class is cloned
    """

    def __init__(self):
        super(MLClassifierBase, self).__init__()

        self.copyable_attrs = []

    def _generate_data_subset(self, y, subset, axis):
        """Subset rows or columns from matrix

        This function subsets the array of binary label vectors to
        include only certain labels.

        Parameters
        ----------
        y : array-like of array-likes
            An array-like of binary label vectors.
        subset: array-like of integers
            array of integers, indices that will be subsetted from
            array-likes in y
        axis: integer 0 for 'rows', 1 for 'labels',
            control variable for whether to return rows or labels as
            indexed by subset

        Returns
        -------
        multi-label binary label vector : array-like of array-likes of {0,1}
            array of binary label vectors including label data only for
            labels from parameter labels
        """
        return_data = None
        if axis == 1:
            return_data = y.tocsc()[:, subset]
        elif axis == 0:
            return_data = y.tocsr()[subset, :]

        return return_data

    def _ensure_input_format(self, X, sparse_format="csr", enforce_sparse=False):
        """Ensure the desired input format

        This function ensures that input format follows the
        density/sparsity requirements of base classifier.

        Parameters
        ----------
        X : array-like or sparse matrix
            An input feature matrix of shape :code:`(n_samples, n_features)`
        sparse_format: str
            Requested format of returned scipy.sparse matrix, if sparse is returned
        enforce_sparse : bool
            Ignore require_dense and enforce sparsity, useful internally

        Returns
        -------
        array-like or sparse matrix
            Transformed X values of shape :code:`(n_samples, n_features)`

        .. note:: If :code:`require_dense` was set to :code:`True` for
            input features in the constructor, the returned value is an
            array-like of array-likes. If :code:`require_dense` is
            set to :code:`false`, a sparse matrix of format
            :code:`sparse_format` is returned, if possible - without cloning.
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

    def _ensure_output_format(self, matrix, sparse_format="csr", enforce_sparse=False):
        """Ensure the desired output format

        This function ensures that output format follows the
        density/sparsity requirements of base classifier.

        Parameters
        ----------

        matrix : array-like matrix
            An input feature matrix of shape :code:`(n_samples)` or
            :code:`(n_samples, n_outputs)` or a sparse matrix of shape
            :code:`(n_samples, n_outputs)`

        sparse_format: str (default is csr)
            Requested format of returned :code:`scipy.sparse` matrix,
            if sparse is returned

        enforce_sparse : bool (default is False)
            Ignore :code:`require_dense` and enforce sparsity, useful
            internally

        Returns
        -------
        array-like or sparse matrix
            Transformed X values of shape :code:`(n_samples, n_features)`

        .. note:: If :code:`require_dense` was set to :code:`True` for
            input features in the constructor, the returned value is an
            array-like of array-likes. If :code:`require_dense` is
            set to :code:`false`, a sparse matrix of format
            :code:`sparse_format` is returned, if possible - without cloning.
        """
        is_sparse = issparse(matrix)

        if is_sparse:
            if self.require_dense[1] and not enforce_sparse:
                if matrix.shape[1] != 1:
                    return matrix.toarray()
                elif matrix.shape[1] == 1:
                    return np.ravel(matrix.toarray())
            else:
                if sparse_format is None:
                    return matrix
                else:
                    return get_matrix_in_format(matrix, sparse_format)
        else:
            if self.require_dense[1] and not enforce_sparse:
                # ensuring 1d
                if len(matrix.shape) > 1:
                    # a regular dense np.matrix or np.array of np.arrays
                    return np.ravel(matrix)
                else:
                    return matrix
            else:
                # ensuring 2d
                if len(matrix.shape) == 1:
                    matrix = matrix.reshape((matrix.shape[0], 1))
                return matrix_creation_function_for_format(sparse_format)(matrix)

    def fit(self, X, y):
        """Abstract method to fit classifier with training data

        It must return a fitted instance of :code:`self`.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        object
            fitted instance of self

        Raises
        ------
        NotImplementedError
            this is just an abstract method
        """

        raise NotImplementedError("MLClassifierBase::fit()")

    def predict(self, X):
        """Abstract method to predict labels

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`

        Raises
        ------
        NotImplementedError
            this is just an abstract method
        """
        raise NotImplementedError("MLClassifierBase::predict()")

    def get_params(self, deep=True):
        """Get parameters to sub-objects

        Introspection of classifier for search models like
        cross-validation and grid search.

        Parameters
        ----------
        deep : bool
            if :code:`True` all params will be introspected also and
            appended to the output dictionary.

        Returns
        -------
        out : dict
            dictionary of all parameters and their values. If
            :code:`deep=True` the dictionary also holds the parameters
            of the parameters.
        """
        out = dict()

        for attr in self.copyable_attrs:
            out[attr] = getattr(self, attr)

            if hasattr(getattr(self, attr), "get_params") and deep:
                deep_items = list(getattr(self, attr).get_params().items())
                out.update((attr + "__" + k, val) for k, val in deep_items)

        return out

    def set_params(self, **parameters):
        """Propagate parameters to sub-objects

        Set parameters as returned by :code:`get_params`. Please
        see this `link`_.

        .. _link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py#L243
        """

        if not parameters:
            return self

        valid_params = self.get_params(deep=True)

        parameters_current_level = [x for x in parameters if "__" not in x]
        for parameter in parameters_current_level:
            value = parameters[parameter]

            if parameter in valid_params:
                setattr(self, parameter, value)
            else:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (parameter, self)
                )

        parameters_below_current_level = [x for x in parameters if "__" in x]
        parameters_grouped_by_current_level = {object: {} for object in valid_params}

        for parameter in parameters_below_current_level:
            object_name, sub_param = parameter.split("__", 1)

            if object_name not in parameters_grouped_by_current_level:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (object_name, self)
                )

            value = parameters[parameter]
            parameters_grouped_by_current_level[object_name][sub_param] = value

        valid_params = self.get_params(deep=True)

        # parameters_grouped_by_current_level groups valid parameters for subojects
        for object_name, sub_params in parameters_grouped_by_current_level.items():
            if len(sub_params) > 0:
                sub_object = valid_params[object_name]
                sub_object.set_params(**sub_params)

        return self
