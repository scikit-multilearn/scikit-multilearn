import math
import numpy as np


def _membership_to_list_of_communities(membership_vector, size):
    """Convert membership vector to list of lists of vertices in each community

    Parameters
    ----------
    membership_vector : list of int
        community membership i.e. vertex/label `i` is in community `membership_vector[i]`
    size : int
        the number of communities present in the membership vector


    Returns
    -------
    list_of_members : list of lists of int
        list of lists of vertex/label ids in each community per community
    """
    list_of_members = [[] for _ in range(size)]
    for vertex_id, community_id in enumerate(membership_vector):
        list_of_members[community_id].append(vertex_id)
    return list_of_members


def _overlapping_membership_to_list_of_communities(membership_vector, size):
    """Convert membership vector to list of lists of vertices/labels in each community

    Parameters
    ----------
    membership_vector : list of lists of int
        community membership i.e. vertex/label `i` is in communities
        from list `membership_vector[i]`
    size : int
        the number of communities present in the membership vector


    Returns
    -------
    list_of_members : list of lists of int
        list of lists of vertex/label ids in each community per community
    """
    list_of_members = [[] for _ in range(size)]
    for vertex_id, community_ids in enumerate(membership_vector):
        for community_id in community_ids:
            list_of_members[community_id].append(vertex_id)
    return list_of_members


def _euclidean_distance(array1, array2):
    """Returns the euclidean distance of two arrays

    Parameters
    ----------
    array1 : array of numbers

    array2 : array of numbers

    Returns
    -------
    distance : float
        float with the euclidean distance, False if not possible
    """
    # Ensure that both arrays hava the same length
    if len(array1) != len(array2):
        return False
    else:
        distance = 0.0
        for i in range(0, len(array1)):
            distance = distance + pow(array1[i] - array2[i], 2)
        distance = math.sqrt(distance)
        return distance


def _recalculateCenters(y, balancedCluster, k):
    Centers = []
    kAux = 0
    while kAux < k:
        vectorAux = np.zeros(len(y))
        for i in range(0, len(balancedCluster)):
            if int(kAux) == int(balancedCluster[i]):
                # We have to fill the vector
                for j in range(0, len(y)):
                    vectorAux[j] += y[j, i]

        vectorAux /= k
        Centers.append(vectorAux)
        kAux += 1
    return Centers


def _countNumberOfAparitions(array, number):
    """Number of aparitions of a number in an array

    Parameters
    ----------
    array : array of numbers

    number : number to search for

    Returns
    -------
    aparaitions : int
        Number of aparitions of the number in the given array
    """
    aparitions = 0
    for i in range(0, len(array)):
        if array[i] == number:
            aparitions += 1
    return aparitions
