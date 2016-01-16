from ..base import MLClassifierBase
import numpy as np

class LabelPowerset(MLClassifierBase):
    """Label Powerset multi-label classifier."""
    BRIEFNAME = "LP"

    def __init__(self, classifier = None):
        super(LabelPowerset, self).__init__(classifier)
        self.clean()

    def clean(self):
        self.unique_combinations = {}
        self.reverse_combinations = []
        self.labelcount = 0

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.clean()
        last_id = 0
        self.labelcount = len(y[0])
        train_vector    = []
        for label_vector in y:
            label_string = str(label_vector)
            if label_string not in self.unique_combinations:
                self.unique_combinations[label_string] = last_id
                self.reverse_combinations.append(label_vector)
                last_id += 1

            train_vector.append(self.unique_combinations[label_string])

        self.classifier.fit(X, train_vector)

        return self

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        lp_prediction = self.classifier.predict(X)

        transformed_to_original_classes = [np.array(self.reverse_combinations[lp_class_id]) for lp_class_id in lp_prediction]
        return transformed_to_original_classes
