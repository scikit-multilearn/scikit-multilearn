from ..base.problem_transformation import ProblemTransformationBase
import numpy as np
from scipy import sparse


class LabelPowerset(ProblemTransformationBase):
    """Transform multi-label problem to a multi-class problem

    Label Powerset is a problem transformation approach to multi-label
    classification that transforms a multi-label problem to a multi-class
    problem with 1 multi-class classifier trained on all unique label
    combinations found in the training data.

    The method maps each combination to a unique combination id number, and performs multi-class classification
    using the `classifier` as multi-class classifier and combination ids as classes.

    Parameters
    ----------
    classifier : :class:`~sklearn.base.BaseEstimator`
        scikit-learn compatible base classifier
    require_dense : [bool, bool], optional
        whether the base classifier requires dense representations
        for input features and classes/labels matrices in fit/predict.
        If value not provided, sparse representations are used if base classifier is
        an instance of :class:`skmultilearn.base.MLClassifierBase` and dense otherwise.

    Attributes
    ----------
    unique_combinations_ : Dict[str, int]
        mapping from label combination as string to label combination id :meth:`transform:` via :meth:`fit`
    reverse_combinations_ : List[List[int]]
        label combination id ordered list to list of label indexes for a given combination  :meth:`transform:`
        via :meth:`fit`

    Notes
    -----
    .. note ::

        `n_classes` in this document denotes the number of unique label combinations present in the training `y`
        passed to :meth:`fit`, in practice it is equal to :code:`len(self.unique_combinations)`

    Examples
    --------
    An example use case for Label Powerset with an :class:`sklearn.ensemble.RandomForestClassifier` base classifier
    which supports sparse input:

    .. code-block:: python

        from skmultilearn.problem_transform import LabelPowerset
        from sklearn.ensemble import RandomForestClassifier

        # initialize LabelPowerset multi-label classifier with a RandomForest
        classifier = ClassifierChain(
            classifier = RandomForestClassifier(n_estimators=100),
            require_dense = [False, True]
        )

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)

    Another way to use this classifier is to select the best scenario from a set of multi-class classifiers used
    with Label Powerset, this can be done using cross validation grid search. In the example below, the model
    with highest accuracy results is selected from either a :class:`sklearn.ensemble.RandomForestClassifier` or
    :class:`sklearn.naive_bayes.MultinomialNB` base classifier, alongside with best parameters for
    that base classifier.

    .. code-block:: python

        from skmultilearn.problem_transform import LabelPowerset
        from sklearn.model_selection import GridSearchCV
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier

        parameters = [
            {
                'classifier': [MultinomialNB()],
                'classifier__alpha': [0.7, 1.0],
            },
            {
                'classifier': [RandomForestClassifier()],
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__n_estimators': [10, 20, 50],
            },
        ]

        clf = GridSearchCV(LabelPowerset(), parameters, scoring='accuracy')
        clf.fit(x, y)

        print (clf.best_params_, clf.best_score_)

        # result
        # {
        #   'classifier': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #             max_depth=None, max_features='auto', max_leaf_nodes=None,
        #             min_impurity_decrease=0.0, min_impurity_split=None,
        #             min_samples_leaf=1, min_samples_split=2,
        #             min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
        #             oob_score=False, random_state=None, verbose=0,
        #             warm_start=False), 'classifier__criterion': 'gini', 'classifier__n_estimators': 50
        # } 0.16

    """

    def __init__(self, classifier=None, require_dense=None):
        super(LabelPowerset, self).__init__(
            classifier=classifier, require_dense=require_dense)
        self._clean()

    def _clean(self):
        """Reset classifier internals before refitting"""
        self.unique_combinations_ = {}
        self.reverse_combinations_ = []
        self._label_count = None

    def fit(self, X, y):
        """Fits classifier to training data

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
        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)

        self.classifier.fit(self._ensure_input_format(X),
                            self.transform(y))

        return self

    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """

        # this will be an np.array of integers representing classes
        lp_prediction = self.classifier.predict(self._ensure_input_format(X))

        return self.inverse_transform(lp_prediction)

    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix

        Returns
        -------
        :mod:`scipy.sparse` matrix of `float in [0.0, 1.0]`, shape=(n_samples, n_labels)
            matrix with label assignment probabilities
        """

        lp_prediction = self.classifier.predict_proba(
            self._ensure_input_format(X))
        result = sparse.lil_matrix(
            (X.shape[0], self._label_count), dtype='float')
        for row in range(len(lp_prediction)):
            assignment = lp_prediction[row]
            for combination_id in range(len(assignment)):
                for label in self.reverse_combinations_[combination_id]:
                    result[row, label] += assignment[combination_id]

        return result

    def transform(self, y):
        """Transform multi-label output space to multi-class

        Transforms a mutli-label problem into a single-label multi-class
        problem where each label combination is a separate class.

        Parameters
        -----------
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        numpy.ndarray of `{0, ... , n_classes-1}`, shape=(n_samples,)
            a multi-class output space vector

        """

        y = self._ensure_output_format(
            y, sparse_format='lil', enforce_sparse=True)

        self._clean()
        self._label_count = y.shape[1]

        last_id = 0
        train_vector = []
        for labels_applied in y.rows:
            label_string = ",".join(map(str, labels_applied))

            if label_string not in self.unique_combinations_:
                self.unique_combinations_[label_string] = last_id
                self.reverse_combinations_.append(labels_applied)
                last_id += 1

            train_vector.append(self.unique_combinations_[label_string])

        return np.array(train_vector)

    def inverse_transform(self, y):
        """Transforms multi-class assignment to multi-label

        Transforms a mutli-label problem into a single-label multi-class
        problem where each label combination is a separate class.

        Parameters
        -----------
        y : numpy.ndarray of `{0, ... , n_classes-1}`, shape=(n_samples,)
            binary indicator matrix with label assignments

        Returns
        -------
        :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        n_samples = len(y)
        result = sparse.lil_matrix((n_samples, self._label_count), dtype='i8')
        for row in range(n_samples):
            assignment = y[row]
            result[row, self.reverse_combinations_[assignment]] = 1

        return result
