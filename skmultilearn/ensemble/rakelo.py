from .voting import MajorityVotingClassifier
from ..cluster.random import RandomLabelSpaceClusterer
from ..problem_transform import LabelPowerset
from ..base import MLClassifierBase


class RakelO(MLClassifierBase):
    """Overlapping RAndom k-labELsets multi-label classifier

    Divides the label space in to m subsets of size k, trains a Label Powerset
    classifier for each subset and assign a label to an instance
    if more than half of all classifiers (majority) from clusters that contain the label
    assigned the label to the instance.

    Parameters
    ----------
    base_classifier: :class:`~sklearn.base.BaseEstimator`
        scikit-learn compatible base classifier, will be set under `self.classifier.classifier`.
    base_classifier_require_dense : [bool, bool]
        whether the base classifier requires [input, output] matrices
        in dense representation. Will be automatically
        set under `self.classifier.require_dense`
    labelset_size : int
        the desired size of each of the partitions, parameter k according to paper.
        According to paper, the best parameter is 3, so it's set as default
        Will be automatically set under `self.labelset_size`
    model_count : int
        the desired number of classifiers, parameter m according to paper.
        According to paper, the best value for this parameter is 2M (being M the number of labels)
        Will be automatically set under :code:`self.model_count_`.


    Attributes
    ----------
    classifier : :class:`~skmultilearn.ensemble.MajorityVotingClassifier`
        the voting classifier initialized with :class:`~skmultilearn.problem_transform.LabelPowerset` multi-label
        classifier with `base_classifier` and :class:`~skmultilearn.cluster.random.RandomLabelSpaceClusterer`


    References
    ----------

    If you use this class please cite the paper introducing the method:

    .. code :: latex

        @ARTICLE{5567103,
            author={G. Tsoumakas and I. Katakis and I. Vlahavas},
            journal={IEEE Transactions on Knowledge and Data Engineering},
            title={Random k-Labelsets for Multilabel Classification},
            year={2011},
            volume={23},
            number={7},
            pages={1079-1089},
            doi={10.1109/TKDE.2010.164},
            ISSN={1041-4347},
            month={July},
        }

    Examples
    --------

    Here's a simple example of how to use this class with a base classifier from scikit-learn to teach 6 classifiers
    each trained on a quarter of labels, which is sure to overlap:

    .. code :: python

        from sklearn.naive_bayes import GaussianNB
        from skmultilearn.ensemble import RakelO

        classifier = RakelO(
            base_classifier=GaussianNB(),
            base_classifier_require_dense=[True, True],
            labelset_size=y_train.shape[1] // 4,
            model_count_=6
        )

        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_train, y_train)

    """

    def __init__(
        self,
        base_classifier=None,
        model_count=None,
        labelset_size=3,
        base_classifier_require_dense=None,
    ):
        super(RakelO, self).__init__()

        self.model_count = model_count
        self.labelset_size = labelset_size
        self.base_classifier = base_classifier
        self.base_classifier_require_dense = base_classifier_require_dense
        self.copyable_attrs = [
            "model_count",
            "labelset_size",
            "base_classifier_require_dense",
            "base_classifier",
        ]

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
        """
        self.classifier = MajorityVotingClassifier(
            classifier=LabelPowerset(
                classifier=self.base_classifier,
                require_dense=self.base_classifier_require_dense,
            ),
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=self.labelset_size,
                cluster_count=self.model_count,
                allow_overlap=True,
            ),
            require_dense=[False, False],
        )
        return self.classifier.fit(X, y)

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

        return self.classifier.predict(X)

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
        return self.classifier.predict_proba(X)
