from builtins import range
from ..base import MLClassifierBase
from ..utils import get_matrix_in_format

from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sparse


class MLkNN(MLClassifierBase):
    """kNN classification method adapted for multi-label classification

    MLkNN builds uses k-NearestNeighbors find nearest examples to a test class and uses Bayesian inference
    to select assigned labels.

    Parameters
    ----------
    k : int
        number of neighbours of each input instance to take into account
    s: float (default is 1.0)
        the smoothing parameter
    ignore_first_neighbours : int (default is 0)
        ability to ignore first N neighbours, useful for comparing
        with other classification software.
    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

    Attributes
    ----------
    knn_ : an instance of sklearn.NearestNeighbors
        the nearest neighbors single-label classifier used underneath



    .. note:: If you don't know what :code:`ignore_first_neighbours`
              does, the default is safe. Please see this `issue`_.


    .. _issue: https://github.com/scikit-multilearn/scikit-multilearn/issues/22


    References
    ----------

    If you use this classifier please cite the original paper introducing the method:

    .. code :: bibtex

        @article{zhang2007ml,
          title={ML-KNN: A lazy learning approach to multi-label learning},
          author={Zhang, Min-Ling and Zhou, Zhi-Hua},
          journal={Pattern recognition},
          volume={40},
          number={7},
          pages={2038--2048},
          year={2007},
          publisher={Elsevier}
        }

    Examples
    --------

    Here's a very simple example of using MLkNN with a fixed number of neighbors:

    .. code :: python

        from skmultilearn.adapt import MLkNN

        classifier = MLkNN(k=3)

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)


    You can also use :class:`~sklearn.model_selection.GridSearchCV` to find an optimal set of parameters:

    .. code :: python

        from skmultilearn.adapt import MLkNN
        from sklearn.model_selection import GridSearchCV

        parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
        score = 'f1_macro'

        clf = GridSearchCV(MLkNN(), parameters, scoring=score)
        clf.fit(X, y)

        print (clf.best_params_, clf.best_score_)

        # output
        ({'k': 1, 's': 0.5}, 0.78988303374297597)

    """

    def __init__(self, k=10, s=1.0, ignore_first_neighbours=0, n_jobs=None):
        """Initializes the classifier

        Parameters
        ----------
        k : int
            number of neighbours of each input instance to take into account
        s: float (default is 1.0)
            the smoothing parameter
        ignore_first_neighbours : int (default is 0)
            ability to ignore first N neighbours, useful for comparing
            with other classification software.
        n_jobs: int or None, optional (default=None)
            The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.


        Attributes
        ----------
        knn_ : an instance of sklearn.NearestNeighbors
            the nearest neighbors single-label classifier used underneath

        .. note:: If you don't know what :code:`ignore_first_neighbours`
                  does, the default is safe. Please see this `issue`_.

        .. _issue: https://github.com/scikit-multilearn/scikit-multilearn/issues/22
        """
        super(MLkNN, self).__init__()
        self.k = k  # Number of neighbours
        self.s = s  # Smooth parameter
        self.ignore_first_neighbours = ignore_first_neighbours
        self.n_jobs = n_jobs
        self.knn_ = NearestNeighbors(n_neighbors=self.k, n_jobs=self.n_jobs)
        self.copyable_attrs = ["k", "s", "ignore_first_neighbours", "n_jobs"]

    def _compute_prior(self, y):
        """Helper function to compute for the prior probabilities

        Parameters
        ----------
        y : numpy.ndarray or scipy.sparse
            the training labels

        Returns
        -------
        numpy.ndarray
            the prior probability given true
        numpy.ndarray
            the prior probability given false
        """
        prior_prob_true = np.array(
            (self.s + y.sum(axis=0)) / (self.s * 2 + self._num_instances)
        )[0]
        prior_prob_false = 1 - prior_prob_true

        return (prior_prob_true, prior_prob_false)

    def _compute_cond(self, X, y):
        """Helper function to compute for the posterior probabilities

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        numpy.ndarray
            the posterior probability given true
        numpy.ndarray
            the posterior probability given false
        """

        self.knn_.fit(X)
        c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")
        cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")

        label_info = get_matrix_in_format(y, "dok")

        neighbors = [
            a[self.ignore_first_neighbours :]
            for a in self.knn_.kneighbors(
                X, self.k + self.ignore_first_neighbours, return_distance=False
            )
        ]

        for instance in range(self._num_instances):
            deltas = label_info[neighbors[instance], :].sum(axis=0)
            for label in range(self._num_labels):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix(
            (self._num_labels, self.k + 1), dtype="float"
        )
        cond_prob_false = sparse.lil_matrix(
            (self._num_labels, self.k + 1), dtype="float"
        )
        for label in range(self._num_labels):
            for neighbor in range(self.k + 1):
                cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                    self.s * (self.k + 1) + c_sum[label, 0]
                )
                cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                    self.s * (self.k + 1) + cn_sum[label, 0]
                )
        return cond_prob_true, cond_prob_false

    def fit(self, X, y):
        """Fit classifier with training data

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        self
            fitted instance of self
        """

        self._label_cache = get_matrix_in_format(y, "lil")
        self._num_instances = self._label_cache.shape[0]
        self._num_labels = self._label_cache.shape[1]
        # Computing the prior probabilities
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(
            self._label_cache
        )
        # Computing the posterior probabilities
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(
            X, self._label_cache
        )
        return self

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse matrix of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype="i8")
        neighbors = [
            a[self.ignore_first_neighbours :]
            for a in self.knn_.kneighbors(
                X, self.k + self.ignore_first_neighbours, return_distance=False
            )
        ]
        for instance in range(X.shape[0]):
            deltas = self._label_cache[neighbors[instance],].sum(axis=0)

            for label in range(self._num_labels):
                p_true = (
                    self._prior_prob_true[label]
                    * self._cond_prob_true[label, deltas[0, label]]
                )
                p_false = (
                    self._prior_prob_false[label]
                    * self._cond_prob_false[label, deltas[0, label]]
                )
                result[instance, label] = int(p_true >= p_false)

        return result

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse matrix of int
            binary indicator matrix with label assignment probabilities
            with shape :code:`(n_samples, n_labels)`
        """
        result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype="float")
        neighbors = [
            a[self.ignore_first_neighbours :]
            for a in self.knn_.kneighbors(
                X, self.k + self.ignore_first_neighbours, return_distance=False
            )
        ]
        for instance in range(X.shape[0]):
            deltas = self._label_cache[neighbors[instance],].sum(axis=0)

            for label in range(self._num_labels):
                p_true = (
                    self._prior_prob_true[label]
                    * self._cond_prob_true[label, deltas[0, label]]
                )
                p_false = (
                    self._prior_prob_false[label]
                    * self._cond_prob_false[label, deltas[0, label]]
                )
                result[instance, label] = p_true / (p_true + p_false)

        return result
