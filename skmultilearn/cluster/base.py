from ..utils import get_matrix_in_format


class LabelSpaceClustererBase(object):
    """docstring for LabelSpaceClustererBase"""

    def __init__(self):
        super(LabelSpaceClustererBase, self).__init__()

    def fit_predict(self, X, y):
        raise NotImplementedError("LabelSpaceClustererBase::fit_predict()")


class LabelCooccurenceClustererBase(LabelSpaceClustererBase):
    """Base class providing API and common functions for all label cooccurence based multi-label classifiers.

    Parameters
    ----------

    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.

    weighted: boolean
            Decide whether to generate a weighted or unweighted graph.

    """

    def __init__(self):
        super(LabelCooccurenceClustererBase, self).__init__()

    def generate_coocurence_adjacency_matrix(self, y):
        """This function generates a weighted or unweighted cooccurence graph based on input binary label vectors 
        and sets it to self.coocurence_graph

        y : array-like of edges
            An array-like of binary label vectors.


        Returns
        -------

        edge_map: dict{ (int, int) : float }
            Returns a dict of weights
        """
        label_data = get_matrix_in_format(y, 'lil')
        self.label_count = label_data.shape[1]

        edge_map = {}

        for row in label_data.rows:
            pairs = [(a, b) for b in row for a in row if a < b]
            for p in pairs:
                if p not in edge_map:
                    edge_map[p] = 1.0
                else:
                    if self.is_weighted:
                        edge_map[p] += 1.0

        self.edge_map = edge_map
        return edge_map
