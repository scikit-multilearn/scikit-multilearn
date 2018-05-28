from builtins import object
from builtins import range

import numpy
import numpy.core.umath as umath
import scipy.sparse
from scipy.sparse import issparse

from ..base import MLClassifierBase


# copyright @Fernando Benites


class Neuron(object):

    def __init__(self, startpoint, label):
        # vector must be in complement form
        self.vc = startpoint
        self.label = label


def _get_label_combination_representation(label_assignment_binary_indicator_list):
    return label_assignment_binary_indicator_list.nonzero()[0].tostring()


def _get_label_vector(y, i):
    if issparse(y):
        return numpy.squeeze(numpy.asarray(y[i].todense()))
    return y[i]


def _normalize_input_space(X):
    x_max = X.max()
    x_min = X.min()
    if x_max < 0 or x_max > 1 or x_min < 0 or x_min > 1:
        return numpy.multiply(X - x_min, 1 / (x_max - x_min))
    return X


class MLARAM(MLClassifierBase):
    """HARAM: A Hierarchical ARAM Neural Network for Large-Scale Text Classification

    This method aims at increasing the classification speed by adding an
    extra ART layer for clustering learned prototypes into large clusters.
    In this case the activation of all prototypes can be replaced by the
    activation of a small fraction of them, leading to a significant
    reduction of the classification time [ICDMW2015]_. 

    Published work available `here`_.

    .. _here: http://dx.doi.org/10.1109/ICDMW.2015.14

    .. [ICDMW2015] F. Benites and E. Sapozhnikova, "HARAM: A Hierarchical
        ARAM Neural Network for Large-Scale Text Classification," 
        2015 IEEE International Conference on Data Mining Workshop
    """
    BRIEFNAME = "ML-ARAM"

    def __init__(self, vigilance=0.9, threshold=0.02, neurons=[]):
        """Initializes the network

        Attributes
        ----------
        vigilance : float (default is 0.9)
            parameter for adaptiv resonance theory networks, controls how
            large a hyperbox can be, 1 it is small (no compression), 0
            should assume all range. Normally set between 0.8 and 0.999,
            it is dataset dependent. It is responsible for the creation
            of the prototypes, therefore training of the network.
        threshold : float (default is 0.02)
            controls how many prototypes participate by the prediction,
            can be changed at the testing phase.
        neurons : list
            the neurons in the network
        """
        super(MLARAM, self).__init__()

        self.neurons = neurons
        self.vigilance = vigilance
        self.threshold = threshold

        self.copyable_attrs += ["neurons", "vigilance", "threshold"]

    def reset(self):
        """Resets the labels and neurons"""
        self.labels = []
        self.neurons = []

    # @profile
    def fit(self, X, y):
        """Fit classifier with training data

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        skmultilearn.MLARAMfast.MLARAM
            fitted instance of self
        """

        self.labels = []
        self.allneu = ""
        self.online = 1
        self.alpha = 0.0000000000001

        label_combination_to_class_map = {}
        is_matrix = int(len(X[0].shape) != 1)
        if isinstance(X, numpy.matrix):
            X = X.toarray()
        X = _normalize_input_space(X)

        y_0 = _get_label_vector(y, 0)
        ones = scipy.ones(X[0].shape)

        if len(self.neurons) == 0:
            if issparse(X):
                X_0 = numpy.squeeze(numpy.asarray(X[0].todense()))
                if is_matrix:
                    neuron_vc = scipy.sparse.hstack((X_0, ones - X_0))
                else:
                    neuron_vc = scipy.sparse.vstack((X_0, ones - X_0))
                self.neurons.append(Neuron(neuron_vc, y_0))
            else:
                self.neurons.append(
                    Neuron(numpy.concatenate((X[0], ones - X[0]), is_matrix), y_0))
            start_index = 1
            label_combination_to_class_map[_get_label_combination_representation(y_0)] = [0]
        else:
            start_index = 0

        # denotes the class enumerator for label combinations
        last_used_label_combination_class_id = 0

        for row_no, input_vector in enumerate(X[start_index:], start_index):
            label_assignment_vector = _get_label_vector(y, row_no)

            if issparse(input_vector):
                input_vector = input_vector.todense()

            fc = numpy.concatenate((input_vector, ones - input_vector), is_matrix)
            activationn = [0] * len(self.neurons)
            activationi = [0] * len(self.neurons)
            label_combination = _get_label_combination_representation(label_assignment_vector)

            if label_combination in label_combination_to_class_map:
                fcs = fc.sum()
                for class_number in label_combination_to_class_map[label_combination]:
                    minnfs = umath.minimum(self.neurons[class_number].vc, fc).sum()
                    activationi[class_number] = minnfs / fcs
                    activationn[class_number] = minnfs / self.neurons[class_number].vc.sum()

            if numpy.max(activationn) == 0:
                last_used_label_combination_class_id += 1
                self.neurons.append(Neuron(fc, label_assignment_vector))
                label_combination_to_class_map.setdefault(label_combination, []).append(len(self.neurons) - 1)

                continue

            inds = numpy.argsort(activationn)
            indc = numpy.where(numpy.array(activationi)[inds[::-1]] > self.vigilance)[0]

            if indc.shape[0] == 0:
                self.neurons.append(Neuron(fc, label_assignment_vector))
                label_combination_to_class_map.setdefault(label_combination, []).append(len(self.neurons) - 1)
                continue

            winner = inds[::- 1][indc[0]]
            self.neurons[winner].vc = umath.minimum(
                self.neurons[winner].vc, fc)

            # 1 if winner neuron won a given label 0 if not
            labels_won_indicator = numpy.zeros(y_0.shape, dtype=y_0.dtype)
            labels_won_indicator[label_assignment_vector.nonzero()] = 1
            self.neurons[winner].label += labels_won_indicator

        return self

    # @profile
    def predict(self, X):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        result = []
        if isinstance(X, numpy.matrix):
            X = X.toarray()
        ranks = self.predict_proba(X)
        for rank in ranks:
            sorted_rank_arg = numpy.argsort(-rank)
            diffs = -numpy.diff([rank[k] for k in sorted_rank_arg])

            indcutt = numpy.where(diffs == diffs.max())[0]
            if len(indcutt.shape) == 1:
                indcut = indcutt[0] + 1
            else:
                indcut = indcutt[0, -1] + 1
            label = numpy.zeros(rank.shape)

            label[sorted_rank_arg[0:indcut]] = 1

            result.append(label)

        return numpy.array(numpy.matrix(result))

    # @profile
    def predict_proba(self, X):
        """Predict probabilities of label assignments for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        array of arrays of float
            matrix with label assignment probabilities of shape
            :code:`(n_samples, n_labels)`
        """
        if isinstance(X, numpy.matrix):
            X = X.toarray()
        if issparse(X):
            if X.getnnz() == 0:
                return
        elif len(X) == 0:
            return

        is_matrix = int(len(X[0].shape) != 1)
        X = _normalize_input_space(X)
        ones = scipy.ones(X[0].shape)
        all_ranks = []
        all_neurons = numpy.vstack([n1.vc for n1 in self.neurons])
        all_neurons_sum = all_neurons.sum(1) + self.alpha

        for row_number, input_vector in enumerate(X):
            if issparse(input_vector):
                input_vector = input_vector.todense()

            fc = numpy.concatenate((input_vector, ones - input_vector), is_matrix)
            activity = (umath.minimum(fc, all_neurons).sum(1) / all_neurons_sum).squeeze().tolist()

            if is_matrix:
                activity = activity[0]

            # be very fast
            sorted_activity = numpy.argsort(activity)[::-1]
            winner = sorted_activity[0]
            activity_difference = activity[winner] - activity[sorted_activity[-1]]
            largest_activity = 1
            par_t = self.threshold

            for i in range(1, len(self.neurons)):
                activity_change = (activity[winner] - activity[sorted_activity[i]]) / activity[winner]
                if activity_change > par_t * activity_difference:
                    break

                largest_activity += 1

            rbsum = sum([activity[k] for k in sorted_activity[0:largest_activity]])
            rank = activity[winner] * self.neurons[winner].label
            activated = []
            activity_among_activated = []
            activated.append(winner)
            activity_among_activated.append(activity[winner])

            for i in range(1, largest_activity):
                rank += activity[sorted_activity[i]] * self.neurons[
                    sorted_activity[i]].label
                activated.append(sorted_activity[i])
                activity_among_activated.append(activity[sorted_activity[i]])

            rank /= rbsum
            all_ranks.append(rank)

        return numpy.array(numpy.matrix(all_ranks))
