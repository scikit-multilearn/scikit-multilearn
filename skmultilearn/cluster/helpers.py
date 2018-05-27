def _membership_to_list_of_communities(membership_vector, size):
    list_of_members = [[] for _ in range(size)]
    for vertex_id, community_id in enumerate(membership_vector):
        list_of_members[community_id].append(vertex_id)
    return list_of_members

def _overlapping_membership_to_list_of_communities(membership_vector, size):
    list_of_members = [[] for _ in range(size)]
    for vertex_id, community_ids in enumerate(membership_vector):
        for community_id in community_ids:
            list_of_members[community_id].append(vertex_id)
    return list_of_members