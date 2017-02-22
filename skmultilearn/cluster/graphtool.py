from __future__ import absolute_import
from builtins import range
from .base import LabelCooccurenceClustererBase
import numpy as np
import graph_tool.all as gt


class GraphToolCooccurenceClusterer(LabelCooccurenceClustererBase):
    """ Clusters the label space using graph tool's stochastic block modelling community detection method

        Parameters
        ----------

        weighted: boolean
                Decide whether to generate a weighted or unweighted graph.

        allow_overlap: boolean
                Allow overlapping of clusters or not.

        include_self_edges : boolean
            Decide whether to include self-edge i.e. label 1 - label 1 in co-occurrence graph

        n_iters : int
                Number of iterations to perform in sweeping 

        n_init_iters: int
                Number of iterations to perform 

        collect_community_statistics: bool
                Whether to collect statistics related to detected community

        use_degree_corr: 'auto' or bool
                Whether to use a degree correlated stochastic blockmodel, or not - if auto, it is selected based on selection criterium

        model_selection_criterium: 'mean_field' or 'bethe'
                Approach to use in case

        verbose: bool
                Be verbose about the output


    """

    def __init__(self, weighted=None, include_self_edges = None, allow_overlap=None):
        super(GraphToolCooccurenceClusterer, self).__init__(weighted=weighted, include_self_edges=include_self_edges)

        self.allow_overlap = allow_overlap
        self.n_iters
        self.n_init_iters
        self.use_degree_corr
        self.model_selection_criterium
        self.collect_community_statistics

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
        self.vm_  = {}
        self.en_  = {}
        self.h_   = {}

        self.generate_coocurence_adjacency_matrix(y)
        self.generate_coocurence_graph()

        d = gt.minimize_blockmodel_dl(
            self.coocurence_graph, overlap=self.allow_overlap) #, ec=self.weights), broken in graphtool for now
        A = d.get_blocks().a

        self.label_sets = [[] for i in range(d.B)]
        for k in range(len(A)):
            self.label_sets[A[k]].append(k)

        self.model_count = len(self.label_sets)

        return np.array(self.label_sets)

    def predict_communities(self, deg_corr):
        if self.is_weighted:
            state = gt.minimize_blockmodel_dl(self.coocurence_graph, overlap=self.allow_overlap,
                deg_corr=deg_corr, layers=True, state_args=dict(ec=self.weights, layers=False))
        else:
            state = gt.minimize_blockmodel_dl(self.coocurence_graph, overlap=self.allow_overlap, deg_corr=deg_corr)
            
        state = state.copy(B=g.num_vertices())

        if self.collect_community_statistics:
            dls = []         # description length history
            vm = None        # vertex marginals
            em = None        # edge marginals
            h = np.zeros(g.num_vertices() + 1)

        def collect_marginals(s):
            global vm, em, dls, h
            vm = s.collect_vertex_marginals(vm)
            em = s.collect_edge_marginals(em)
            dls.append(s.entropy())
            B = s.get_nonempty_B()
            h[B] += 1

        # Now we collect the marginal distributions for exactly 200,000 sweeps
        if self.collect_community_statistics:
            gt.mcmc_equilibrate(state, force_niter=n_iters, mcmc_args=dict(niter=10),
                            callback=collect_marginals)
            
            self.dls_[deg_corr] = copy.copy(dls)
            self.vm_[deg_corr] = copy.copy(vm)
            self.en_[deg_corr] = copy.copy(em)
            self.h_[deg_corr] =copy.copy(h)

        else:
            gt.mcmc_equilibrate(state, force_niter=n_iters, mcmc_args=dict(niter=10))



        S_mf = gt.mf_entropy(g, vm)
        S_bethe = gt.bethe_entropy(g, em)[0]
        L = -np.mean(dls)

        self.state_[deg_corr] =copy.copy(state)
        self.S_bethe_[deg_corr] =copy.copy(S_bethe)
        self.S_mf_[deg_corr] =copy.copy(S_mf)
        self.L_[deg_corr] =copy.copy(L)

        print("Model evidence for deg_corr = %s:" % deg_corr,
              L + S_mf, "(mean field),", L + S_bethe, "(Bethe)")
