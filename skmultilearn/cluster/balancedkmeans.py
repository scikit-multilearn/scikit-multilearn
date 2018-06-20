from __future__ import absolute_import

import numpy as np
import random

from .base import LabelSpaceClustererBase

class balancedKMeansClusterer(LabelSpaceClustererBase):
    "Balanced clustering algorithm which is used by HOMER"
    def __init__(self, k = None, it = None):
        super(balancedKMeansClusterer, self).__init__()
        self.k = k
        self.it = it

    def fit_predict(self, X, y):
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
                    distances.append(euclidean_distance(v, Centers[i]))
                finished = False
                while not finished:
                    minIndex = np.argmin(distances)
                    balancedCluster.append(minIndex)
                    #Now we have the cluster we want to add this label to
                    numberOfAparitions = countNumberOfAparitions(balancedCluster, minIndex)
                    if float(numberOfAparitions) > (float(float(number_of_labels)/float(self.k))+1):
                        distances[minIndex] = float("inf")
                        balancedCluster.pop()
                    else:
                        finished = True

                
            Centers = recalculateCenters(np.asarray(y), balancedCluster, self.k)
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