from builtins import zip
from builtins import map
from builtins import range

class RandomOrderedClassifierChain(MLClassifierBase):
    """Classifier Chains multi-label classifier."""
    BRIEFNAME = "CC"

    def __init__(self, classifier=None):
        super(RandomOrderedClassifierChain, self).__init__(classifier)
        self.ordering = None

    def generate_label_ordering(self):
        self.ordering = random.sample(
            range(self.label_count), self.label_count)

    def fit(self, X, y):
        # fit L = len(y[0]) BR classifiers h_i
        # on X + y[:i] as input space and y[i+1] as output
        #
        self.predictions = y
        self.num_instances = len(y)
        self.label_count = len(y[0])
        self.classifiers = [None for x in range(self.label_count)]
        self.draw_ordering()

        for label in range(self.label_count):
            classifier = copy.deepcopy(self.classifier)
            y_tolearn = self.generate_data_subset(y, self.ordering[label])
            y_toinput = self.generate_data_subset(y, self.ordering[:label])

            X_extended = np.append(X, y_toinput, axis=1)
            classifier.fit(X_extended, y_tolearn)
            self.classifiers[self.ordering[label]] = classifier

        return self

    def predict(self, X):
        result = np.zeros((len(X), self.label_count), dtype='i8')
        for instance in range(len(X)):
            predictions = []
            for label in self.ordering:
                prediction = self.classifiers[label].predict(
                    np.append(X[instance], predictions))
                predictions.append(prediction)
                result[instance][label] = prediction
        return result


class EnsembleClassifierChains(MLClassifierBase):
    """docstring for EnsembleClassifierChains"""

    def __init__(self,
                 classifier=None,
                 model_count=None,
                 training_sample_percentage=None,
                 threshold=None):
        super(EnsembleClassifierChains, self).__init__(classifier)
        self.model_count = model_count
        self.threshold = threshold
        self.percentage = training_sample_percentage
        self.models = None

    def fit(self, X, y):
        self.models = []
        self.label_count = len(y[0])
        for model in range(self.model_count):
            base_classifier = copy.deepcopy(self.classifier)
            classifier = RandomOrderedClassifierChain(base_classifier)
            sampled_rows = random.sample(
                range(len(X)), int(self.percentage * len(X)))
            classifier.fit(self.generate_data_subset(
                X, sampled_rows, 'rows'), self.generate_data_subset(y, sampled_rows, 'rows'))
            self.models.append(classifier)
        return self

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        predictions = [
            self.ensure_input_format(self.ensure_input_format(
                c.predict(X)), sparse_format='csc', enforce_sparse=True)
            for c in self.classifiers
        ]

        votes = sparse.csc_matrix(
            (predictions[0].shape[0], self.label_count), dtype='i8')
        for model in range(self.model_count):
            for label in range(len(self.partition[model])):
                votes[:, self.partition[model][label]] = votes[
                    :, self.partition[model][label]] + predictions[model][:, label]

        voters = list(map(float, votes.sum(axis=0).tolist()[0]))

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            if votes[row, column] >= self.threshold * voters[column]:
                votes[row, column] = 1

        return votes
