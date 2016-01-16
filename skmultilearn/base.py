import numpy as np

class MLClassifierBase(object):
    """Base class providing API and common functions for all multi-label classifiers.

    Parameters
    ----------

    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.
    """
    def __init__(self, classifier = None):

        super(MLClassifierBase, self).__init__()
        self.classifier = classifier

    def generate_data_subset(self, y, subset, axis = 'labels'):
        """This function subsets the array of binary label vectors to include only certain labels. 

        Parameters
        ----------

        y : array-like of array-likes
            An array-like of binary label vectors.

        subset: array-like of integers
            array of integers, indices that will be subsetted from array-likes in y

        axis: enum{'labels', 'rows'}
            control variable for whether to return rows or labels as indexed by subset

        Returns
        -------

        multi-label binary label vector : array-like of array-likes of {0,1}
            array of binary label vectors including label data only for labels from parameter labels
        """
        if axis == 'labels':
            return [row[subset] for row in y]
        elif axis == 'rows':
            return [y[i] for i in subset]
        else:
            return None

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

        out["classifier"] = self.classifier

        # deep introspection of estimator parameters
        if deep and hasattr(self.classifier, 'get_params'):
            deep_items = value.get_params().items()
            out.update((key + '__' + k, val) for k, val in deep_items)

        return out

    def set_params(self, **parameters):
        """
        Set parameters as returned by `get_params`.

        Parameters
        ----------

        parameters : dict
            Dictionary of parameters as returned by `get_params`. Sets each of the
            classifiers parameters to the ones as given by the dictionary

        """

        if not params:
            return self

        for parameter, value in parameters.items():
            self.setattr(parameter, value)

        return self
