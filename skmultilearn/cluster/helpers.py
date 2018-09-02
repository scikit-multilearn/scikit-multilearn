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