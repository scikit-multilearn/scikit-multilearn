# -*- coding: utf-8 -*-
import numpy as np
import itertools as it


def example_distribution(folds, desired_size):
    """Examples Distribution (ED) measure

     Examples Distribution is a measure of how much a given fold's size deviates from the desired number
     of samples in each of the folds.

    Parameters:
    -----------
    folds : List[List[int]], shape = (n_folds)
        list of indexes of samples assigned per fold

    desired_size : List[int], shape = (n_folds)
        desired number of samples in each fold

    Returns
    -------
    example_distribution_score : float

        The example distribution score
    """
    n_splits = float(len(folds))

    return np.sum(
        np.abs(len(fold) - desired_fold_size) for fold, desired_fold_size in zip(folds, desired_size)
    ) / n_splits


def get_indicator_representation(row):
    """Convert binary indicator to list of assigned labels

    Parameters:
    -----------

    row : List[{0,1}]
        binary indicator list whether i-th label is assigned or not

    Returns
    -------
    np.array[int]
        list of assigned labels
    """
    return np.where(row != 0)[0]


def get_combination_wise_output_matrix(y, order):
    """Returns label combinations of a given order that are assigned to each row

    Parameters:
    -----------
    y : output matrix or array of arrays (n_samples, n_labels)
        the binary-indicator label assignment per sample representation of the output space

    order : int, >= 1
        the order of label relationship to take into account when balancing sample distribution across labels

    Returns
    -------
    combinations_per_row : List[Set[Tuple[int]]]
        list of combination assignments per row
    """
    return np.array([set(tuple(combination) for combination in
                         it.combinations_with_replacement(get_indicator_representation(row), order)) for row in y])


def get_unique_combinations(combinations_per_row):
    """Performs set.union on a list of sets

    Parameters
    ----------

    combinations_per_row : List[Set[Tuple[int]]]
        list of combination assignments per row

    Returns
    -------
    Set[Tuple[int]]
        all unique label combinations
    """
    return set.union(*combinations_per_row)


def folds_without_evidence_for_at_least_one_label_combination(y, folds, order=1):
    """Counts the number of folds without evidence for a given Label, Label Pair or Label Combination (FZ, FZLP, FZLC) measure

    A general implementation of FZ - the number of folds that contain at least one label combination of order
    :code:`order` with no positive examples. With :code:`order` = 1, it becomes the FZ measure from Katakis et.al's
    original paper.

    Parameters:
    -----------
    y : output matrix or array of arrays (n_samples, n_labels)
        the binary-indicator label assignment per sample representation of the output space

    folds : List[List[int]], shape = (n_folds)
        list of indexes of samples assigned per fold

    order : int, >= 1
        the order of label relationship to take into account when balancing sample distribution across labels

    Returns
    -------
    score : float
        the number of folds with missing evidence for at least one label combination
    """
    combinations_per_row = get_combination_wise_output_matrix(y, order)
    all_combinations = get_unique_combinations(combinations_per_row)
    return np.sum([get_unique_combinations(combinations_per_row[[fold]]) != all_combinations for fold in folds])


def folds_label_combination_pairs_without_evidence(y, folds, order):
    """Fold - Label / Label Pair / Label Combination (FLZ, FLPZ, FLCZ)  pair count measure

    A general implementation of FLZ - the number of pairs of fold and label combination of a given order for which
    there is no positive evidence in that fold for that combination. With :code:`order` = 1, it becomes the FLZ
    measure from Katakis et.al's original paper, with :code:`order` = 2, it becomes the FLPZ measure from
    Szymański et. al.'s paper.

    Parameters:
    -----------
    y : output matrix or array of arrays (n_samples, n_labels)
        the binary-indicator label assignment per sample representation of the output space

    folds : List[List[int]], shape = (n_folds)
        list of indexes of samples assigned per fold

    order : int, >= 1
        the order of label relationship to take into account when balancing sample distribution across labels

    Returns
    -------
    score : float
        the number of fold-label combination pairs with missing evidence
    """
    combinations_per_row = get_combination_wise_output_matrix(y, order)
    all_combinations = get_unique_combinations(combinations_per_row)
    return np.sum(
        [len(all_combinations.difference(get_unique_combinations(combinations_per_row[[fold]]))) for fold in folds])


def percentage_of_label_combinations_without_evidence_per_fold(y, folds, order):
    """Fold - Label / Label Pair / Label Combination (FLZ, FLPZ, FLCZ)  pair count measure

    A general implementation of FLZ - the number of pairs of fold and label combination of a given order for which
    there is no positive evidence in that fold for that combination. With :code:`order` = 1, it becomes the FLZ
    measure from Katakis et.al's original paper, with :code:`order` = 2, it becomes the FLPZ measure from
    Szymański et. al.'s paper.

    Parameters:
    -----------
    y : output matrix or array of arrays (n_samples, n_labels)
        the binary-indicator label assignment per sample representation of the output space

    folds : List[List[int]], shape = (n_folds)
        list of indexes of samples assigned per fold

    order : int, >= 1
        the order of label relationship to take into account when balancing sample distribution across labels

    Returns
    -------
    score : float
        the number of fold-label combination pairs with missing evidence
    """
    combinations_per_row = get_combination_wise_output_matrix(y, order)
    all_combinations = get_unique_combinations(combinations_per_row)
    number_of_combinations = float(len(all_combinations))
    return [
        1.0 - len(get_unique_combinations(combinations_per_row[[fold]])) / number_of_combinations for fold in folds
    ]


def label_combination_distribution(y, folds, order):
    """Label / Label Pair / Label Combination Distribution (LD, LPD, LCZD) measure

    A general implementation of Label / Label Pair / Label Combination Distribution - a measure that evaluates
    how the proportion of positive evidence for a label / label pair / label combination to the negative evidence
    for a label (pair/combination) deviates from the same proportion in the entire data set, averaged over all folds and labels.

    With :code:`order` = 1, it becomes the LD measure from Katakis et.al's original paper, with :code:`order` = 2, it
    becomes the LPD measure from Szymański et. al.'s paper.

    Parameters:
    -----------
    y : output matrix or array of arrays (n_samples, n_labels)
        the binary-indicator label assignment per sample representation of the output space

    folds : List[List[int]], shape = (n_folds)
        list of indexes of samples assigned per fold

    order : int, >= 1
        the order of label relationship to take into account when balancing sample distribution across labels

    Returns
    -------
    score : float
        the label / label pair / label combination distribution score
    """

    def _get_proportion(x, y):
        return y / float(x - y)

    combinations_per_row = get_combination_wise_output_matrix(y, order)
    all_combinations = get_unique_combinations(combinations_per_row)
    number_of_samples = y.shape[0]
    number_of_combinations = float(len(all_combinations))
    number_of_folds = float(len(folds))

    external_sum = 0
    for combination in all_combinations:
        number_of_samples_with_combination = np.sum([
            1 for combinations_in_row in combinations_per_row if combination in combinations_in_row
        ])

        d = _get_proportion(number_of_samples, number_of_samples_with_combination)
        internal_sum = 0
        for fold in folds:
            S_i_j = np.sum(
                [1 for combinations_in_row in combinations_per_row[fold] if combination in combinations_in_row])
            fold_size = len(fold)
            s = _get_proportion(fold_size, S_i_j)
            internal_sum += np.abs(s - d)

        internal_sum /= number_of_folds
        external_sum += internal_sum

    return external_sum / number_of_combinations
