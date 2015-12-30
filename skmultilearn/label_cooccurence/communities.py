from .base import LabelCooccurenceClassifier
from ..base import RepeatClassifier
import copy
import numpy as np

from scipy import sparse
from ..utils import get_matrix_in_format

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
    
        self.community_count = len(self.label_sets)
        return self

    def fit_only(self, X, y):
        self.classifiers = []

        for i in xrange(self.community_count):
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
        input_rows = X.shape[0]
        predictions = [self.classifiers[i].predict(X) for i in xrange(self.community_count)]
        result = np.zeros((input_rows, self.label_count))

        for row in xrange(input_rows):
            for model in xrange(self.community_count):
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
        self.params = dict()


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
        self.community_count = len(self.label_sets)

        assert self.label_count is not None

        self.community = np.zeros(self.label_count, dtype=int)

        for community_no in xrange(self.community_count):
            for label in self.label_sets[community_no]:
                self.community[label] = community_no

        return self

    def includes_a_class_spanning_over_all_objects(self, y):
        testdata = get_matrix_in_format(y,'csc')
        all_rows = y.shape[0]
        all_columns = y.shape[1]

        for i in xrange(all_columns):
            nonzero_value_count = testdata[:,i].nnz 
            if nonzero_value_count == all_rows or nonzero_value_count == 0:
                return True
        return False

    def fit_only(self, X, y, escape_one_class = True):
        assert self.community_count is not None
        # we might have only one community, that's fine
        if self.community_count == 1:
            self.level1_classifier = copy.deepcopy(self.classifier)
            return self.level1_classifier.fit(X, y)

        # level1
        input_rows = y.shape[0]
        level1_y = sparse.lil_matrix((input_rows, self.community_count), dtype=int)

        y_rows = get_matrix_in_format(y,'lil').rows

        for row in xrange(len(y_rows)):
            for column in y_rows[row]:
                level1_y[row, self.community[column]] = 1

        self.level1_classifier = copy.deepcopy(self.classifier)
        if escape_one_class and self.includes_a_class_spanning_over_all_objects(level1_y):
            # return the same class for everyone?
            self.level1_classifier = None
            self.level1_class = level1_y[0,:].toarray()[0]
        else:
            self.level1_classifier.fit(X, get_matrix_in_format(level1_y, 'csr'))

        y_csc = get_matrix_in_format(y, 'csc')
        X_csr = get_matrix_in_format(X, 'csr')

        self.classifiers = []
        for community in xrange(self.community_count):
            y_per_community = y_csc[:,self.label_sets[community]]
            X_with_labels_from_community = np.unique(y_per_community.nonzero()[0])

            if len(X_with_labels_from_community) == 0:
                # we have no rows classified into this community, so blast it into oblivion
                classifier = None
                self.classifiers.append(classifier)
            else:
                classifier = None
                y_per_community_csr = get_matrix_in_format(y_per_community,'csr')
                y_subsetted = y_per_community_csr[X_with_labels_from_community,:]
                needs_escaping = escape_one_class and self.includes_a_class_spanning_over_all_objects(y_subsetted)
                # now check how many labels we have in community
                if not needs_escaping:
                    classifier = copy.deepcopy(self.classifier)
                    classifier.fit(X_csr[X_with_labels_from_community,:], y_subsetted)
                else:
                    classifier = copy.deepcopy(self.classifier)
                    classifier.fit(X, y_per_community)

                self.classifiers.append(classifier)
        return self
        

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.generate_clustering(y)
        
        return self.fit_only(X,y)

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        if self.level1_classifier is None:
            # handle a case when all observations have to be checked in all subclassifiers
            assert self.level1_class is not None

            input_rows = X.shape[0]
            result = sparse.lil_matrix((input_rows, self.label_count), dtype=int)

            for model in xrange(self.community_count):
                if self.level1_class[model] == 0:
                    continue

                level2_predictions = self.classifiers[model].predict(X)

                for label_position in xrange(len(self.label_sets[model])):
                    for row in xrange(X.shape[0]):
                        if level2_predictions[row, label_position] == 1:
                            result[row, self.label_sets[model][label_position]] = 1

            return result



        level1_predictions = self.level1_classifier.predict(X)
        if self.community_count == 1:
            return level1_predictions

        self.level1_predictions = level1_predictions
        input_rows = X.shape[0]
        result = sparse.lil_matrix((input_rows, self.label_count), dtype=int)
        row_ids = [[] for model in xrange(self.community_count)]

        non_zero_positions = level1_predictions.nonzero()

        for row, column in zip(non_zero_positions[0], non_zero_positions[1]):
            row_ids[column].append(row)
            
        for model in xrange(self.community_count):
            if len(row_ids[model]) == 0:
                continue
            if self.classifiers[model] is None:
                # this should only happen in one-class models of length 1
                assert len(self.label_sets[model]) == 1
                level2_predictions = level1_predictions[self.label_sets[model], row_ids[model]]
            else:
                level2_predictions = self.classifiers[model].predict(X[row_ids[model]])

            for label_position in xrange(len(self.label_sets[model])):
                for row in xrange(len(row_ids[model])):
                    if level2_predictions[row, label_position] == 1:
                        result[row_ids[model][row], self.label_sets[model][label_position]] = 1

        return result

    def get_params(deps = True):
        return self.params

    def set_params(**params):
        self.params = params
        self.classifier.set_params(params)
        if self.level1_classifier is not None:
            self.level1_classifier.set_params(params)

        for clf in self.classifiers:
            if cls is not None:
                clf.set_params(params)

        return self