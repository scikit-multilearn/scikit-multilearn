from ..base import MLClassifierBase
import numpy as np
from scipy import sparse

class LabelPowerset(MLClassifierBase):
    """Label Powerset multi-label classifier."""
    BRIEFNAME = "LP"
    
    def __init__(self, classifier = None, require_dense = None):
        super(LabelPowerset, self).__init__(classifier = classifier, require_dense = require_dense)
        self.clean()

    def clean(self):
        self.unique_combinations = {}
        self.reverse_combinations = []
        self.label_count = None

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        X = self.ensure_input_format(X, sparse_format = 'csr', enforce_sparse = True)
        y = self.ensure_output_format(y, sparse_format = 'lil', enforce_sparse = True)
        self.clean()
        self.label_count = y.shape[1]
        last_id = 0
        train_vector    = []
        for labels_applied in y_lil.rows:
            label_string = ",".join(map(str,labels_applied))

            if label_string not in self.unique_combinations:
                self.unique_combinations[label_string] = last_id
                self.reverse_combinations.append(labels_applied)
                last_id += 1

            train_vector.append(self.unique_combinations[label_string])

        self.classifier.fit(self.ensure_input_format(X), train_vector)

        return self

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        # this will be an np.array of integers representing classes
        lp_prediction = self.classifier.predict(self.ensure_input_format(X))
        result = sparse.lil_matrix((X.shape[0], self.label_count), dtype='i8')

        for row in xrange(len(lp_prediction)):
            assignment = lp_prediction[row]
            result[row, self.reverse_combinations[assignment]] = 1

        return result

    def transform(self, y):
        """ Transform the label set to a multi-class problem """
        return map(lambda x: int("".join(map(str,x))),y)

    def inverse_transform(self, y):
        return map(lambda x: map(int, str(x)),y)

    def set_params(self, **params):
        if self.classifier is not None:
            self.classifier.set_params(params)

        return self

    def get_params(self, deep = False):
        if deep and self.classifier is not None:
            return self.classifier.get_params()

        return dict()