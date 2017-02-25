from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from .base import LabelCooccurenceClustererBase
import copy
import numpy as np
import graph_tool.all as gt


class GraphToolCooccurenceClusterer(LabelCooccurenceClustererBase):

    """ Clusters the label space using graph tool's stochastic block modelling community detection method

        Parameters
        ----------

        weighted: boolean
                Decide whether to generate a weighted or unweighted graph.

        include_self_edges : boolean
            Decide whether to include self-edge i.e. label 1 - label 1 in co-occurrence graph

        allow_overlap: boolean
                Allow overlapping of clusters or not.

        n_iters : int
                Number of iterations to perform in sweeping

        n_init_iters: int
                Number of iterations to perform

        use_degree_corr: None or bool
                Whether to use a degree correlated stochastic blockmodel, or not - if None, it is selected based on selection criterium

        model_selection_criterium: 'mean_field' or 'bethe'
                Approach to use in case

        verbose: bool
                Be verbose about the output

        equlibrate_options: dict
                additional options to pass to `graphtool's mcmc_equilibrate <https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.mcmc_equilibrate>`_


    """

    def __init__(self, weighted=None,
                 include_self_edges=None,
                 allow_overlap=None,
                 n_iters=100,
                 n_init_iters=10,
                 use_degree_corr=None,
                 model_selection_criterium='mean_field',
                 verbose=False,
                 equlibrate_options={}):
        super(GraphToolCooccurenceClusterer, self).__init__(
            weighted=weighted, include_self_edges=include_self_edges)

        self.allow_overlap = allow_overlap
        self.n_iters = n_iters
        self.n_init_iters = n_init_iters
        self.use_degree_corr = use_degree_corr
        self.model_selection_criterium = model_selection_criterium
        self.verbose = verbose
        self.equlibrate_options = equlibrate_options

        if allow_overlap not in [True, False]:
            raise ValueError("allow_overlap needs to be a boolean")

    def generate_coocurence_graph(self):
        """ Constructs the label coocurence graph

        This function constructs a graph-tool :py:class:`graphtool.Graph` object representing the label cooccurence graph. Run after self.edge_map has been populated using :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix` on `y` in `fit_predict`.

        The graph is available as self.coocurence_graph, and a weight `double` graphtool.PropertyMap on edges is set as self.weights.

        Edge weights are all 1.0 if self.weighted is false, otherwise they contain the number of samples that are labelled with the two labels present in the edge.

        Returns
        -------

        g : graphtool.Graph object representing a label co-occurence graph

        """
        g = gt.Graph(directed=False)
        g.add_vertex(self.label_count)

        self.weights = g.new_edge_property('double')

        for edge, weight in self.edge_map.items():
            e = g.add_edge(edge[0], edge[1])
            if self.is_weighted:
                self.weights[e] = weight
            else:
                self.weights[e] = 1.0

        self.coocurence_graph = g

        return g

    def fit_predict(self, X, y):
        """ Performs clustering on y and returns list of label lists

        Builds a label coocurence_graph using :func:`LabelCooccurenceClustererBase.generate_coocurence_adjacency_matrix` on `y` and then detects communities using graph tool's stochastic block modeling.

        Parameters
        ----------
        X : sparse matrix (n_samples, n_features), feature space, not used in this clusterer
        y : sparse matrix (n_samples, n_labels), label space

        Returns
        -------
        partition: list of lists : list of lists label indexes, each sublist represents labels that are in that community

        """
        self.dls_ = {}
        self.vm_ = {}
        self.em_ = {}
        self.h_ = {}
        self.state_ = {}
        self.S_bethe_ = {}
        self.S_mf_ = {}
        self.L_ = {}

        self.generate_coocurence_adjacency_matrix(y)
        self.generate_coocurence_graph()

        which_model_to_use = None

        if self.use_degree_corr is None:
            for deg_corr in [True, False]:
                self.predict_communities(deg_corr)

            decision_criterion = self.S_mf_

            if self.model_selection_criterium == 'bethe':
                decision_criterion = self.S_bethe_

            which_model_to_use = decision_criterion[
                True] < decision_criterion[False]

        else:
            self.predict_communities(self.use_degree_corr)
            which_model_to_use = self.use_degree_corr

        state_to_use = self.state_[which_model_to_use]

        found_blocks = state_to_use.get_blocks().get_array()

        self.label_sets = {b: [] for b in range(len(found_blocks))}

        for label_id, block_id in enumerate(found_blocks):
            self.label_sets[block_id].append(label_id)

        self.label_sets = filter(
            lambda x: len(x) > 0, self.label_sets.values())

        self.model_count = len(self.label_sets)

        return np.array(self.label_sets)

    def predict_communities(self, deg_corr):
        if self.is_weighted:
            state = gt.minimize_blockmodel_dl(
                self.coocurence_graph, overlap=self.allow_overlap,
                deg_corr=deg_corr, layers=True, state_args=dict(ec=self.weights, layers=False))
        else:
            state = gt.minimize_blockmodel_dl(
                self.coocurence_graph, overlap=self.allow_overlap, deg_corr=deg_corr)

        state = state.copy(B=self.coocurence_graph.num_vertices())

        self.dls_[deg_corr] = []          # description length history
        self.vm_[deg_corr] = None        # vertex marginals
        self.em_[deg_corr] = None        # edge marginals
        self.h_[deg_corr] = np.zeros(
            self.coocurence_graph.num_vertices() + 1)

        def collect_marginals(s, deg_corr, obj):
            obj.vm_[deg_corr] = s.collect_vertex_marginals(obj.vm_[deg_corr])
            obj.em_[deg_corr] = s.collect_edge_marginals(obj.em_[deg_corr])
            obj.dls_[deg_corr].append(s.entropy())
            B = s.get_nonempty_B()
            obj.h_[deg_corr][B] += 1

        collect_marginals_for_class = lambda s: collect_marginals(
            s, deg_corr, self)

        # Now we collect the marginal distributions for exactly 200,000 sweeps
        gt.mcmc_equilibrate(
            state, force_niter=self.n_iters, mcmc_args=dict(
                niter=self.n_init_iters),
                        callback=collect_marginals_for_class, **self.equlibrate_options)

        S_mf = gt.mf_entropy(self.coocurence_graph, self.vm_[deg_corr])
        S_bethe = gt.bethe_entropy(
            self.coocurence_graph, self.em_[deg_corr])[0]
        L = -np.mean(self.dls_[deg_corr])

        self.state_[deg_corr] = copy.copy(state)
        self.S_bethe_[deg_corr] = copy.copy(S_bethe)
        self.S_mf_[deg_corr] = copy.copy(S_mf)
        self.L_[deg_corr] = copy.copy(L)

        if self.verbose:
            print(("Model evidence for deg_corr = %s:" % deg_corr,
                  L + S_mf, "(mean field),", L + S_bethe, "(Bethe)"))
