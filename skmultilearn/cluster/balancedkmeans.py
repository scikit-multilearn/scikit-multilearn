from __future__ import absolute_import

import numpy as np
import random

from .base import LabelSpaceClustererBase
from .helpers import _euclidean_distance, _recalculateCenters, _countNumberOfAparitions

class BalancedKMeansClusterer(LabelSpaceClustererBase):
    """Cluster the label space regarding the algorithm of balancedkMeans, used by HOMER"""

    def __init__(self, k = None, it = None):
        """Initializes the clusterer

        Attributes
        ----------
        k: int
                Number of partitions to be made to the label-space
        it: int
                Number of iterations for the algorithm to find the best neighbours
        """
        super(BalancedKMeansClusterer, self).__init__()
        self.k = k
        self.it = it

    def fit_predict(self, X, y):
        """Performs clustering on y and returns list of label lists

        Builds a label list taking care of the distance between labels

        Parameters
        ----------
        X : currently unused, left for scikit compatibility
        y : scipy.sparse
            label space of shape :code:`(n_samples, n_labels)`

        Returns
        -------
        array of arrays
            numpy array of arrays of label indexes, where each sub-array
            represents labels that are in a separate community
        """
        number_of_labels = y.shape[1]
        #Assign a label to a cluster no. label ordinal %  number of labeSls
        #We have to do the balance k-means and then use it for HOMER with the label powerset
        Centers =[]
        y = y.todense()
        for i in range(0, self.k):
            auxVector = y[:, random.randint(0, number_of_labels-1)]
            Centers.append(np.asarray(auxVector))
        #Now we have the clusters created and we need to make each label its corresponding cluster
        
        while self.it > 0:
            balancedCluster = []
            for j in range(0, number_of_labels):
                auxVector = y[:,j]
                v = np.asarray(auxVector)
                #Now we calculate the distance and store it in an array
                distances = []
                for i in range(0, self.k):
                    #Store the distances
                    distances.append(_euclidean_distance(v, Centers[i]))
                finished = False
                while not finished:
                    minIndex = np.argmin(distances)
                    balancedCluster.append(minIndex)
                    #Now we have the cluster we want to add this label to
                    numberOfAparitions = _countNumberOfAparitions(balancedCluster, minIndex)
                    if float(numberOfAparitions) > (float(float(number_of_labels)/float(self.k))+1):
                        distances[minIndex] = float("inf")
                        balancedCluster.pop()
                    else:
                        finished = True

                
            Centers = _recalculateCenters(np.asarray(y), balancedCluster, self.k)
            self.it = self.it -1

        #Returns a list of list with the clusterers
        labelCluster = []
        for i in range(0, self.k):
            cluster = []
            for j in range(0, len(balancedCluster)):
                if int(i) == int(balancedCluster[j]):
                    cluster.append(int(j))
            labelCluster.append(cluster)

        return np.asarray(labelCluster)