from ..base.problem_transformation import ProblemTransformationBase
from scipy.sparse import hstack
from sklearn.exceptions import NotFittedError
import copy


class ClassifierChain(ProblemTransformationBase):
    """Constructs a bayesian conditioned chain of per label classifiers

    This class provides implementation of Jesse Read's problem
    transformation method called Classifier Chains. For L labels it
    trains L classifiers ordered in a chain according to the
    `Bayesian chain rule`.

    The first classifier is trained just on the input space, and then
    each next classifier is trained on the input space and all previous
    classifiers in the chain.

    The default classifier chains follow the same ordering as provided
    in the training set, i.e. label in column 0, then 1, etc.

    Parameters
    ----------
    classifier : :class:`~sklearn.base.BaseEstimator`
        scikit-learn compatible base classifier
    require_dense : [bool, bool], optional
        whether the base classifier requires dense representations
        for input features and classes/labels matrices in fit/predict.
        If value not provided, sparse representations are used if base classifier is
        an instance of :class:`~skmultilearn.base.MLClassifierBase` and dense otherwise.
    order : List[int], permutation of ``range(n_labels)``, optional
        the order in which the chain should go through labels, the default is ``range(n_labels)``


    Attributes
    ----------
    classifiers_ : List[:class:`~sklearn.base.BaseEstimator`] of shape `n_labels`
        list of classifiers trained per partition, set in :meth:`fit`



    References
    ----------
    If used, please cite the scikit-multilearn library and the relevant paper:

    .. code-block:: bibtex

        @inproceedings{read2009classifier,
          title={Classifier chains for multi-label classification},
          author={Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe},
          booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
          pages={254--269},
          year={2009},
          organization={Springer}
        }

    Examples
    --------
    An example use case for Classifier Chains
    with an :class:`sklearn.svm.SVC` base classifier which supports sparse input:

    .. code-block:: python

        from skmultilearn.problem_transform import ClassifierChain
        from sklearn.svm import SVC

        # initialize Classifier Chain multi-label classifier
        # with an SVM classifier
        # SVM in scikit only supports the X matrix in sparse representation

        classifier = ClassifierChain(
            classifier = SVC(),
            require_dense = [False, True]
        )

        # train
        classifier.fit(X_train, y_train)

        # predict
        predictions = classifier.predict(X_test)

    Another way to use this classifier is to select the best scenario from a set of single-label classifiers used
    with Classifier Chain, this can be done using cross validation grid search. In the example below, the model
    with highest accuracy results is selected from either a :class:`sklearn.naive_bayes.MultinomialNB` or
    :class:`sklearn.svm.SVC` base classifier, alongside with best parameters for that base classifier.

    .. code-block:: python

        from skmultilearn.problem_transform import ClassifierChain
        from sklearn.model_selection import GridSearchCV
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import SVC

        parameters = [
            {
                'classifier': [MultinomialNB()],
                'classifier__alpha': [0.7, 1.0],
            },
            {
                'classifier': [SVC()],
                'classifier__kernel': ['rbf', 'linear'],
            },
        ]


        clf = GridSearchCV(ClassifierChain(), parameters, scoring='accuracy')
        clf.fit(x, y)

        print (clf.best_params_, clf.best_score_)

        # result
        # {'classifier': MultinomialNB(alpha=0.7, class_prior=None, fit_prior=True), 'classifier__alpha': 0.7} 0.16

    """

    def __init__(self, classifier=None, require_dense=None, order=None):
        super(ClassifierChain, self).__init__(classifier, require_dense)
        self.order = order
        self.copyable_attrs = ['classifier', 'require_dense', 'order']

    def fit(self, X, y, order=None):
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

        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output

        X_extended = self._ensure_input_format(X, sparse_format='csc', enforce_sparse=True)
        y = self._ensure_output_format(y, sparse_format='csc', enforce_sparse=True)

        self._label_count = y.shape[1]
        self.classifiers_ = [None for x in range(self._label_count)]

        for label in self._order():
            self.classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, label, axis=1)

            self.classifiers_[label] = self.classifier.fit(self._ensure_input_format(
                X_extended), self._ensure_output_format(y_subset))
            X_extended = hstack([X_extended, y_subset])

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

        X_extended = self._ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)

        for label in self._order():
            prediction = self.classifiers_[label].predict(
                self._ensure_input_format(X_extended))
            prediction = self._ensure_multi_label_from_single_class(prediction)
            X_extended = hstack([X_extended, prediction])
        return X_extended[:, -self._label_count:]

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
        X_extended = self._ensure_input_format(
            X, sparse_format='csc', enforce_sparse=True)

        results = []
        for label in self._order():
            prediction = self.classifiers_[label].predict(
                self._ensure_input_format(X_extended))

            prediction = self._ensure_output_format(
                prediction, sparse_format='csc', enforce_sparse=True)

            prediction_proba = self.classifiers_[label].predict_proba(
                self._ensure_input_format(X_extended))

            prediction_proba = self._ensure_output_format(
                prediction_proba, sparse_format='csc', enforce_sparse=True)[:, 1]

            X_extended = hstack([X_extended, prediction]).tocsc()
            results.append(prediction_proba)

        return hstack(results)

    def _order(self):
        if self.order is not None:
            return self.order

        try:
            return list(range(self._label_count))
        except AttributeError:
            raise NotFittedError("This Classifier Chain has not been fit yet")
