import copy
import numpy as np
from   .utils import get_matrix_in_format


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
            return y[:,subset]
        elif axis == 'rows':
            return y[subset,:]
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

class RepeatClassifier(MLClassifierBase):

    def __init__(self, value_to_repeat = None):
        
        super(RepeatClassifier, self).__init__()

    def fit(self, X, y):
        self.value_to_repeat = copy.copy(y.tocsr[0,:])
        self.return_value = np.full(y)
        return self

    def predict(self, X):
        return np.array([np.copy(self.value_to_repeat) for x in xrange(len(X))])
