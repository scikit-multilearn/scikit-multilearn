import pytest

from skmultilearn.cluster import GraphToolCooccurenceClusterer
from skmultilearn.cluster.base import LabelCooccurenceGraphBuilder
from skmultilearn.cluster.graphtool import StochasticBlockModel
from skmultilearn.tests.example import EXAMPLE_X, EXAMPLE_y

import scipy.sparse as sparse

def get_graphtool_partitioners():
    for nested in [True, False]:
        for degree_correlation in [True, False]:
            for weight_model in [None, 'real-exponential', 'real-normal',
                                 'discrete-geometric', 'discrete-binomial',
                                 'discrete-poisson']:
                sbm = StochasticBlockModel(nested, degree_correlation, False, weight_model)
                bld = LabelCooccurenceGraphBuilder(weighted=weight_model is not None,
                                                   include_self_edges=False,
                                                   normalize_self_edges=False)
                clf = GraphToolCooccurenceClusterer(graph_builder=bld, model=sbm)
                yield clf

@pytest.mark.parametrize("nested,degree_correlation,allow_overlap,weight_model", [
    (True, True, True, None),
    (True, True, True, 'real-exponential'),
    (True, True, True, 'real-normal'),
    (True, True, True, 'discrete-geometric'),
    (True, True, True, 'discrete-binomial'),
    (True, True, True, 'discrete-poisson'),
    (True, True, False, None),
    (True, True, False, 'real-exponential'),
    (True, True, False, 'real-normal'),
    (True, True, False, 'discrete-geometric'),
    (True, True, False, 'discrete-binomial'),
    (True, True, False, 'discrete-poisson'),
    (True, False, False, None),
    (True, False, False, 'real-exponential'),
    (True, False, False, 'real-normal'),
    (True, False, False, 'discrete-geometric'),
    (True, False, False, 'discrete-binomial'),
    (True, False, False, 'discrete-poisson'),
    (False, False, False, None),
    (False, False, False, 'real-exponential'),
    (False, False, False, 'real-normal'),
    (False, False, False, 'discrete-geometric'),
    (False, False, False, 'discrete-binomial'),
    (False, False, False, 'discrete-poisson')
])
def test_that_graph_tool_clusterer_works(nested, degree_correlation, allow_overlap, weight_model):
    sbm = StochasticBlockModel(nested, degree_correlation, allow_overlap, weight_model)
    bld = LabelCooccurenceGraphBuilder(weighted=True, include_self_edges=False, normalize_self_edges=False)
    clf = GraphToolCooccurenceClusterer(graph_builder=bld, model=sbm)
    X, y = sparse.csr_matrix(EXAMPLE_X), sparse.csr_matrix(EXAMPLE_y)
    division = clf.fit_predict(X, y)
    for label in range(y.shape[1]):
        assert any(label in partition for partition in division)
