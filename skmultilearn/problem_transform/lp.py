from builtins import map
from builtins import str
from builtins import range
from ..base.problem_transformation import ProblemTransformationBase
import numpy as np
from scipy import sparse


class LabelPowerset(ProblemTransformationBase):
    """Label Powerset Multi-Label Classifier.

    Label Powerset is a problem transformation approach to multi-label
    classification that transforms a multi-label problem to a multi-class
    problem with 1 multi-class classifier trained on all unique label
    combinations found in the training data.

    More information about this method can be found in an introduction
    to multi-label classification by Tsoumakas et. al.
    """
    BRIEFNAME = "LP"

    def __init__(self, classifier=None, require_dense=None):
        """Initializes the LabelPowerset class

        Attributes
        ----------
        classifier : sklear.base.BaseEstimator
            scikit-compatible base classifier
        require_dense : list of bools ([bool, bool])
            whether the base classifier requires dense representations
            for input features and classes/labels matrices in fit/predict.
        """
        super(LabelPowerset, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self.clean()

    def clean(self):
        """Reset classifier internals before refitting"""
        self.unique_combinations = {}
        self.reverse_combinations = []
        self.label_count = None

    def fit(self, X, y):
        """Fit classifier with training data

        Internally this method uses a sparse CSR representation
        (:class:`scipy.sparse.csr_matrix`) of the X matrix and
        a sparse LIL representation (:class:`scipy.sparse.lil_matrix`).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        skmultilearn.problem_transform.lp.LabelPowerset
            fitted instance of self

        """
        X = self.ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)

        self.classifier.fit(self.ensure_input_format(X),
                            self.transform(y))

        return self

    def predict(self, X):
        """Predict labels for X

        Internally this method uses a sparse CSR representation for X
        (:class:`scipy.sparse.csr_matrix`).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        # this will be an np.array of integers representing classes
        lp_prediction = self.classifier.predict(self.ensure_input_format(X))

        return self.inverse_transform(lp_prediction)

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Internally this method uses a sparse CSR representation for X
        (:class:`scipy.sparse.coo_matrix`).

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of float
            matrix with label assignment probabilities of shape
            :code:`(n_samples, n_labels)`
        """

        lp_prediction = self.classifier.predict_proba(
            self.ensure_input_format(X))
        result = sparse.lil_matrix(
            (X.shape[0], self.label_count), dtype='float')
        for row in range(len(lp_prediction)):
            assignment = lp_prediction[row]
            for combination_id in range(len(assignment)):
                for label in self.reverse_combinations[combination_id]:
                    result[row, label] += assignment[combination_id]

        return result

    def transform(self, y):
        """Transform multi-label output space to multi-class

        Transforms a mutli-label problem into a single-label multi-class
        problem where each label combination is a separate class.

        Parameters
        -----------
        y : numpy.ndarray or scipy.sparse
            output space of shape :code:`(n_samples, n_labels)`
            of {0,1} of a multi-label classification problem

        Returns
        -------
        numpy.ndarray
            a multi-class output space of size :code:`(n_samples, )`

        """

        y = self.ensure_output_format(
            y, sparse_format='lil', enforce_sparse=True)

        self.clean()
        self.label_count = y.shape[1]

        last_id = 0
        train_vector = []
        for labels_applied in y.rows:
            label_string = ",".join(map(str, labels_applied))

            if label_string not in self.unique_combinations:
                self.unique_combinations[label_string] = last_id
                self.reverse_combinations.append(labels_applied)
                last_id += 1

            train_vector.append(self.unique_combinations[label_string])

        return np.array(train_vector)

    def inverse_transform(self, y):
        """Transforms multi-class assignment to multi-label

        Transforms a mutli-label problem into a single-label multi-class
        problem where each label combination is a separate class.

        Parameters
        ----------
        y : numpy.ndarray
            output space of size :code:`(n_samples,)` as transformed by
            :meth:`transform`
        
        Returns
        -------
        array
            assignments following the label combinations of the original
            multi-label classification problem. Binary matrix of shape
            :code:`(n_samples, n_labels)`
        """
        n_samples = len(y)
        result = sparse.lil_matrix((n_samples, self.label_count), dtype='i8')
        for row in range(n_samples):
            assignment = y[row]
            result[row, self.reverse_combinations[assignment]] = 1

        return result
