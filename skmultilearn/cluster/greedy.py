from __future__ import absolute_import

import random

import numpy as np

from .base import LabelSpaceClustererBase

def label_correlation(y, s=0.35):
    """Correlation between labels in a label matrix
    Parameters
    ----------
    y : array-like (n_labels, n_samples)
        Label matrix
    s : float
        Smoothness parameter
    Returns
    -------
    L : array-like (n_labels, n_labels)
        Label correlation matrix
    """
    L = np.zeros(shape=[y.shape[0], y.shape[0]])

    for i in range(0, y.shape[0]):
        yi = sum(y[i,:])
        for j in range(0, y.shape[0]):
            coincidence = 0
            for k in range(0, y.shape[1]):
                if (int(y[i,k]) == int(1)) and (int(y[j,k]) == int(1)):
                    coincidence += 1
            L[i,j] = (coincidence + s)/(yi + 2*s)

    return L

class GreedyLabelSpaceClusterer(LabelSpaceClustererBase):
    """Greedy divides the label space into equally-sized clusters

    This method divides the label space by drawing without replacement a desired number of
    equally sized subsets of label space, in a partitioning or overlapping scheme.

    Parameters
    ----------
    cluster_size : int
        desired size of a single cluster, will be automatically
        put under :code:`self.cluster_size`.
    cluster_count: int
        number of clusters to divide into, will be automatically
        put under :code:`self.cluster_count`.
    allow_overlap : bool
        whether to allow overlapping clusters or not, will be automatically
        put under :code:`self.allow_overlap`.

    Examples
    --------

    The following code performs greedy label space partitioning.

    .. code :: python

        from skmultilearn.cluster import GreedyLabelSpaceClusterer

        # assume X,y contain the data, example y contains 5 labels
        cluster_count = 2
        cluster_size = y.shape[1]//cluster_count # == 2
        clr = GreedyLabelSpaceClusterer(cluster_size, cluster_count, allow_overlap=False)
        clr.fit_predict(X,y)
        # Result:
        # array([list([0, 4]), list([2, 3]), list([1])], dtype=object)


    Note that the leftover labels that did not fit in `cluster_size` x `cluster_count` classifiers will be appended
    to an additional last cluster of size at most `cluster_size` - 1.

    You can also use this class to get a greedy division of the label space, even with multiple overlaps:

    .. code :: python

        from skmultilearn.cluster import GreedyLabelSpaceClusterer

        cluster_size = 3
        cluster_count = 5
        clr = GreedyLabelSpaceClusterer(cluster_size, cluster_count, allow_overlap=True)
        clr.fit_predict(X,y)

        # Result
        # array([[2, 1, 3],
        #        [3, 0, 4],
        #        [2, 3, 1],
        #        [2, 3, 4],
        #        [3, 4, 0],
        #        [3, 0, 2]])


    Note that you will never get the same label subset twice.
    """

    def __init__(self, cluster_size, cluster_count, allow_overlap):
        super(GreedyLabelSpaceClusterer, self).__init__()

        self.cluster_size = cluster_size
        self.cluster_count = cluster_count
        self.allow_overlap = allow_overlap

    def fit_predict(self, X, y):
        """Cluster the output space

        Parameters
        ----------
        X : currently unused, left for scikit compatibility
        y : scipy.sparse
            label space of shape :code:`(n_samples, n_labels)`

        Returns
        -------
        arrray of arrays of label indexes (numpy.ndarray)
            label space division, each sublist represents labels that are in that community
        """

        if (self.cluster_count+1) * self.cluster_size < y.shape[1]:
            raise ValueError("Cannot include all of {} labels in {} clusters of {} labels".format(
                y.shape[1],
                self.cluster_count,
                self.cluster_size
            ))

        all_labels_assigned_to_division = False
        # make sure the final label set division includes all labels
        while not all_labels_assigned_to_division:
            label_sets = []
            free_labels = range(y.shape[1])

            while len(label_sets) <= self.cluster_count:
                if not self.allow_overlap:
                    if len(free_labels) == 0:
                        break

                    # in this case, we are unable to draw new labels, add all that remain
                    if len(free_labels) < self.cluster_size:
                        label_sets.append(free_labels)
                        break

                #Changing the random way into a greedy way, it takes all of the labels by random sampling
                #Takes a random label
                    #Greedily interacts with each of the labels draw from the labelset
                    #Creates the label_subset and then continues
                L = label_correlation(np.transpose(y))
                L = np.transpose(L)
                #Get a random label
                first_label = random.sample(free_labels, 1)
                label_set = np.zeros(self.cluster_size)
                label_set[0] = first_label
                for i in range(1, len(label_set)):
                    pass_next = False
                    while not pass_next:
                        next_label = np.argmax(L[:,label_set[i-1]])
                        if next_label not in free_labels and next_label not in label_set:
                            pass_next = True
                    label_set[i] = next_label

                if not self.allow_overlap:
                    free_labels = list(set(free_labels).difference(set(label_set)))

                if label_set not in label_sets:
                    label_sets.append(label_set)

            all_labels_assigned_to_division = all(
                any(label in subset for subset in label_sets)
                for label in range(y.shape[1])
            )

        return np.array(label_sets)