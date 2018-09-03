# copyright @Fernando Benites
from builtins import object
from builtins import range

import numpy
import numpy.core.umath as umath
import scipy.sparse
from scipy.sparse import issparse

from ..base import MLClassifierBase


class Neuron(object):
    """An implementation of a neuron for MLARAM

    Parameters
    ----------
    vc : array
        neuron's assigned vector
    label : int
        label number
    """

    def __init__(self, vc, label):
        # vector must be in complement form
        self.vc = vc
        self.label = label


def _get_label_combination_representation(label_assignment_binary_indicator_list):
    return label_assignment_binary_indicator_list.nonzero()[0].tostring()


def _get_label_vector(y, i):
    if issparse(y):
        return numpy.squeeze(numpy.asarray(y[i].todense()))
    return y[i]

def _concatenate_with_negation(row):
    ones = scipy.ones(row.shape)
    if issparse(row):
        return scipy.sparse.hstack((row, ones - row))
    else:
        # concatenate and merge sublists in the row if it is matrix
        return numpy.concatenate((row, ones - row), int(len(row.shape) != 1))

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
    reduction of the classification time.

    Parameters
    ----------
    vigilance : float (default is 0.9)
        parameter for adaptive resonance theory networks, controls how
        large a hyperbox can be, 1 it is small (no compression), 0
        should assume all range. Normally set between 0.8 and 0.999,
        it is dataset dependent. It is responsible for the creation
        of the prototypes, therefore training of the network.
    threshold : float (default is 0.02)
        controls how many prototypes participate by the prediction,
        can be changed for the testing phase.
    neurons : list
        the neurons in the network


    References
    ----------

    Published work available `here`_.

    .. _here: http://dx.doi.org/10.1109/ICDMW.2015.14

    .. code :: bibtex

        @INPROCEEDINGS{7395756,
            author={F. Benites and E. Sapozhnikova},
            booktitle={2015 IEEE International Conference on Data Mining Workshop (ICDMW)},
            title={HARAM: A Hierarchical ARAM Neural Network for Large-Scale Text Classification},
            year={2015},
            volume={},
            number={},
            pages={847-854},
            doi={10.1109/ICDMW.2015.14},
            ISSN={2375-9259},
            month={Nov},
        }

    Examples
    --------

    Here's an example code with a 5% threshold and vigilance of 0.95:

    .. code :: python

        from skmultilearn.neurofuzzy import MLARAM

        classifier = MLARAM(threshold=0.05, vigilance=0.95)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)


    """

    def __init__(self, vigilance=0.9, threshold=0.02, neurons=None):
        super(MLARAM, self).__init__()

        if neurons is not None:
            self.neurons = neurons
        else:
            self.neurons = []
        self.vigilance = vigilance
        self.threshold = threshold

        self.copyable_attrs += ["neurons", "vigilance", "threshold"]

    def reset(self):
        """Resets the labels and neurons"""
        self._labels = []
        self.neurons = []

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

        self._labels = []
        self._allneu = ""
        self._online = 1
        self._alpha = 0.0000000000001

        is_sparse_x = issparse(X)

        label_combination_to_class_map = {}
        # FIXME: we should support dense matrices natively
        if isinstance(X, numpy.matrix):
            X = numpy.asarray(X)
        if isinstance(y, numpy.matrix):
            y = numpy.asarray(y)
        is_more_dimensional = int(len(X[0].shape) != 1)
        X = _normalize_input_space(X)

        y_0 = _get_label_vector(y, 0)

        if len(self.neurons) == 0:
            neuron_vc = _concatenate_with_negation(X[0])
            self.neurons.append(Neuron(neuron_vc, y_0))
            start_index = 1
            label_combination_to_class_map[_get_label_combination_representation(y_0)] = [0]
        else:
            start_index = 0

        # denotes the class enumerator for label combinations
        last_used_label_combination_class_id = 0

        for row_no, input_vector in enumerate(X[start_index:], start_index):
            label_assignment_vector = _get_label_vector(y, row_no)

            fc = _concatenate_with_negation(input_vector)
            activationn = [0] * len(self.neurons)
            activationi = [0] * len(self.neurons)
            label_combination = _get_label_combination_representation(label_assignment_vector)

            if label_combination in label_combination_to_class_map:
                fcs = fc.sum()
                for class_number in label_combination_to_class_map[label_combination]:
                    if issparse(self.neurons[class_number].vc):
                        minnfs = self.neurons[class_number].vc.minimum(fc).sum()
                    else:
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
            if issparse(self.neurons[winner].vc):
                self.neurons[winner].vc = self.neurons[winner].vc.minimum(fc)
            else:
                self.neurons[winner].vc = umath.minimum(
                    self.neurons[winner].vc, fc
                )

            # 1 if winner neuron won a given label 0 if not
            labels_won_indicator = numpy.zeros(y_0.shape, dtype=y_0.dtype)
            labels_won_indicator[label_assignment_vector.nonzero()] = 1
            self.neurons[winner].label += labels_won_indicator

        return self

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
        # FIXME: we should support dense matrices natively
        if isinstance(X, numpy.matrix):
            X = numpy.asarray(X)
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
        # FIXME: we should support dense matrices natively
        if isinstance(X, numpy.matrix):
            X = numpy.asarray(X)
        if issparse(X):
            if X.getnnz() == 0:
                return
        elif len(X) == 0:
            return

        is_matrix = int(len(X[0].shape) != 1)
        X = _normalize_input_space(X)

        all_ranks = []
        neuron_vectors = [n1.vc for n1 in self.neurons]
        if any(map(issparse, neuron_vectors)):
            all_neurons = scipy.sparse.vstack(neuron_vectors)
            # can't add a constant to a sparse matrix in scipy
            all_neurons_sum = all_neurons.sum(1).A
        else:
            all_neurons = numpy.vstack(neuron_vectors)
            all_neurons_sum = all_neurons.sum(1)

        all_neurons_sum += self._alpha

        for row_number, input_vector in enumerate(X):
            fc = _concatenate_with_negation(input_vector)

            if issparse(fc):
                activity = (fc.minimum(all_neurons).sum(1) / all_neurons_sum).squeeze().tolist()
            else:
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
