class LabelPowerset(object):
    """docstring for LabelPowerset"""

    def __init__(self, classifier=None):
        super(LabelPowerset, self).__init__()
        self.classifier = classifier
        self.clean()

    def clean(self):
        self.unique_combinations = {}
        self.reverse_combinations = []

    def fit(self, X, y):
        self.clean()
        last_id = 0
        train_vector = []
        for label_vector in y:
            label_string = str(label_vector)
            if label_string not in self.unique_combinations:
                self.unique_combinations[label_string] = self.last_id
                self.reverse_combinations.append(label_string)
                last_id += 1

            train_vector.append(self.unique_combinations[label_string])

        self.classifier.fit(X, train_vector)

        return self


    def predict(self, X):
        lp_prediction = self.classifier.predict(X)
        transformed_to_original_classes = [eval(self.reverse_combinations[lp_class_id]) for lp_class_id in lp_prediction]
        return transformed_to_original_classes
