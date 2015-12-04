from .base import LabelCooccurenceClassifier
from ..base import RepeatClassifier
import copy
import numpy as np

class LabelDistinctCommunitiesClassifier(LabelCooccurenceClassifier):
    """Community detection base classifier

    Parameters
    ----------

    classifier : scikit classifier type
    The base classifier that will be used in a class, will be automagically put under self.classifier for future access.
    
    community_detection_method: a function that returns an array-like of array-likes of integers for a given igraph.Graph 
    and weights vector
        An igraph.Graph object and a weight vector (if weighted == True) will be passed to this function expecting a return 
        of a division of graph nodes into communities represented array-like of array-likes or vector containing label 
        ids per community


    weighted: boolean
            Decide whether to generate a weighted or unweighted co-occurence graph.

    """
    def __init__(self, classifier = None, weighted = None, community_detection_method = None):
        super(LabelDistinctCommunitiesClassifier, self).__init__(classifier, weighted)
        self.community_detection_method = community_detection_method


    def generate_clustering(self, y):
        self.generate_coocurence_graph(y)
        self.label_sets = None

        if self.is_weighted:
            self.label_sets = self.community_detection_method(self.coocurence_graph, self.weights)
        else:
            self.label_sets = self.community_detection_method(self.coocurence_graph)
    
        self.model_count = len(self.label_sets)
        return self

    def fit_only(self, X, y):
        self.classifiers = []

        for i in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y,self.label_sets[i])
            classifier.fit(X,y_subset)
            self.classifiers.append(classifier)
        return self
        

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.generate_clustering(y)
        return self.fit_only(X,y)

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        input_rows = len(X)
        predictions = [self.classifiers[i].predict(X) for i in xrange(self.model_count)]
        result = np.zeros((input_rows, self.label_count))

        for row in xrange(input_rows):
            for model in xrange(self.model_count):
                for label_position in xrange(len(self.label_sets[model])):
                    if predictions[model][row][label_position] == 1:
                        label_id = self.label_sets[model][label_position]
                        result[row][label_id] = 1
        return result


class HierarchicalDistinctCommunitiesClassifier(LabelCooccurenceClassifier):
    """Community detection base classifier

    Parameters
    ----------

    classifier : scikit classifier type
    The base classifier that will be used in a class, will be automagically put under self.classifier for future access.
    
    community_detection_method: a function that returns an array-like of array-likes of integers for a given igraph.Graph 
    and weights vector
        An igraph.Graph object and a weight vector (if weighted == True) will be passed to this function expecting a return 
        of a division of graph nodes into communities represented array-like of array-likes or vector containing label 
        ids per community


    weighted: boolean
            Decide whether to generate a weighted or unweighted co-occurence graph.

    """
    def __init__(self, classifier = None, weighted = None, community_detection_method = None):
        super(HierarchicalDistinctCommunitiesClassifier, self).__init__(classifier, weighted)
        self.community_detection_method = community_detection_method


    def generate_clustering(self, y):
        self.generate_coocurence_graph(y)
        self.label_sets = None

        if self.is_weighted:
            self.label_sets = self.community_detection_method(self.coocurence_graph, self.weights)
        else:
            self.label_sets = self.community_detection_method(self.coocurence_graph)

        singletons = reduce(lambda x,y: x+y, filter(lambda x: len(x) == 1, self.label_sets), [])
        self.label_sets = filter(lambda x: len(x) > 1, self.label_sets)
        if len(singletons) > 0:
            self.label_sets.append(singletons)
        self.model_count = len(self.label_sets)

        return self

    def has_more_than_one_class(self, y):
        found_value = y[0]
        for i in xrange(1, len(y)):
            if not np.array_equal(y[i], found_value):
                return True
        return False

    def fit_only(self, X, y, escape_one_class = True):
        # we might have only one community, that's fine
        if len(self.label_sets) == 1:
            self.level1_classifier = copy.deepcopy(self.classifier)
            return self.level1_classifier.fit(X, y)

        # level1 
        input_rows = len(y)
        level1_y = np.zeros((input_rows, self.model_count))
        row_ids = [[] for i in xrange(self.model_count)]

        # for each row in train data
        for level2_row_no in xrange(len(y)):
            # make label row for metalabels
            level1_row = [0 for i in xrange(self.model_count)]
            for i in xrange(self.model_count):
                for l in self.label_sets[i]:
                    # if label is attached, i.e. > 0
                    if y[level2_row_no][l] > 0:
                        level1_y[level2_row_no][i] = 1
                        row_ids[i].append(level2_row_no)
                        break

        self.level1_classifier = copy.deepcopy(self.classifier)
        if escape_one_class and not self.has_more_than_one_class(level1_y):
            # return the same class for everyone?
            self.level1_classifier = RepeatClassifier(level1_y[0])
        else:
            self.level1_classifier.fit(X, level1_y)

        self.classifiers = []
        for i in xrange(self.model_count):
            if len(row_ids[i]) == 0:
                # we have no rows classified into this community, so blast it into oblivion
                classifier = None
                self.classifiers.append(classifier)
            else:
                classifier = copy.deepcopy(self.classifier)
                if len(self.label_sets[i]) > 1:
                    y_subset   = self.generate_data_subset(y[row_ids[i]], self.label_sets[i])
                    if escape_one_class and not self.has_more_than_one_class(y_subset):
                        classifier = None
                    else:
                        classifier.fit(X[row_ids[i]], y_subset)
                else:
                    y_subset   = self.generate_data_subset(y, self.label_sets[i])
                    if escape_one_class and not self.has_more_than_one_class(y_subset):
                        classifier = None
                    else:
                        classifier.fit(X, y_subset)

                self.classifiers.append(classifier)
        return self
        

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.generate_clustering(y)
        
        return self.fit_only(X,y)

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        level1_predictions = np.array(self.level1_classifier.predict(X))
        if len(self.label_sets) == 1:
            return level1_predictions

        self.level1_predictions=level1_predictions
        input_rows = len(X)
        result = np.zeros((input_rows, self.label_count))
        row_ids = [None for model in xrange(self.model_count)]

        for model in xrange(self.model_count):
            row_ids[model] = filter(lambda x: level1_predictions[x][model] > 0, xrange(input_rows))
            if len(row_ids[model]) == 0:
                continue
            if self.classifiers[model] is None:
                # this should only happen in one-class models of length 1
                assert len(self.label_sets[model]) == 1
                level2_predictions = level1_predictions[self.label_sets[model]][row_ids[model]]
            else:
                level2_predictions = self.classifiers[model].predict(X[row_ids[model]])
            for label_position in xrange(len(self.label_sets[model])):
                for row in xrange(len(row_ids[model])):
                    if level2_predictions[row][label_position] == 1:
                        result[row_ids[model][row]][self.label_sets[model][label_position]] = 1

        return result
