from .base import LabelCooccurenceClassifier
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
            self.label_sets = self.community_detection_method(self.coocurence_graph, weights = self.weights)
        else:
            self.label_sets = self.community_detection_method(self.coocurence_graph)
    
        self.model_count = len(self.label_sets)
        return self
        

    def fit(self, X, y):
        """Fit classifier according to X,y, see base method's documentation."""
        self.generate_clustering(y)
        self.classifiers = []

        for i in xrange(self.model_count):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self.generate_data_subset(y,self.label_sets[i])
            classifier.fit(X,y_subset)
            self.classifiers.append(classifier)
        return self


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