from ..meta.br import BinaryRelevance
import copy
import numpy as np

from scipy import sparse
from ..utils import get_matrix_in_format

class LabelSpacePartitioningClassifier(BinaryRelevance):
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
    def __init__(self, classifier = None, clusterer = None, require_dense = None):
        super(LabelDistinctCommunitiesClassifier, self).__init__(classifier, require_dense)
        self.clusterer = clusterer


    def generate_partition(self, X, y):
        self.partition = self.clusterer.fit(X, y)  
        self.model_count = len(self.partition)
        
        return self

    def predict(self, X):
        """Predict labels for X, see base method's documentation."""
        result = sparse.lil_matrix((input_rows, self.model_count), dtype=int)

        for model in xrange(self.model_count):
            predictions = self.ensure_output_format(self.classifiers[model].predict(X), sparse_format = None, enforce_sparse = True).nonzero()
            for row, column in zip(predictions[0], predictions[1]):
                result[row, self.partition[model][column]] = 1

        return result
