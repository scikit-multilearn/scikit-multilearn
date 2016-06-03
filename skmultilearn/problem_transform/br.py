from ..base.problem_transformation import ProblemTransformationBase
from scipy.sparse import hstack, coo_matrix
from sklearn.utils import check_array
import copy


class BinaryRelevance(ProblemTransformationBase):
    """Binary Relevance multi-label classifier."""
    BRIEFNAME = "BR"

    def __init__(self, classifier=None, require_dense=None):
        super(BinaryRelevance, self).__init__(classifier, require_dense)

    def generate_partition(self, X, y):
        self.partition = range(y.shape[1])
        self.model_count = y.shape[1]

    def fit(self, X, y):
        """Fit classifier according to `X`, `y`, see base method's documentation."""
        X = self.ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self.ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.generate_partition(X, y)
        self.classifiers = []

        for i in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y, self.partition[i], axis=1)
            classifier.fit(self.ensure_input_format(
                X), self.ensure_output_format(y_subset))
            self.classifiers.append(classifier)

        return self

    def predict(self, X):
        """Predict labels for `X`, see base method's documentation."""
        predictions = [self.classifiers[label].predict(
            self.ensure_input_format(X)) for label in xrange(self.model_count)]
        if isinstance(self.classifier, ProblemTransformationBase):
            return hstack(predictions)
        else:
            return coo_matrix(predictions).T
