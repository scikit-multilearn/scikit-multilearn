import numpy as np
import scipy.sparse as sp

SPARSE_FORMAT_TO_CONSTRUCTOR = {
    "bsr": sp.bsr_matrix,
    "coo": sp.coo_matrix,
    "csc": sp.csc_matrix,
    "csr": sp.csr_matrix,
    "dia": sp.dia_matrix,
    "dok": sp.dok_matrix,
    "lil": sp.lil_matrix,
}


def get_matrix_in_format(original_matrix, matrix_format):
    """Converts matrix to format

    Parameters
    ----------

    original_matrix : np.matrix or scipy matrix or np.array of np. arrays
        matrix to convert

    matrix_format : string
        format

    Returns
    -------

    matrix : scipy matrix
        matrix in given format
    """
    if isinstance(original_matrix, np.ndarray):
        return SPARSE_FORMAT_TO_CONSTRUCTOR[matrix_format](original_matrix)

    if original_matrix.getformat() == matrix_format:
        return original_matrix

    return original_matrix.asformat(matrix_format)


def matrix_creation_function_for_format(sparse_format):
    if sparse_format not in SPARSE_FORMAT_TO_CONSTRUCTOR:
        return None

    return SPARSE_FORMAT_TO_CONSTRUCTOR[sparse_format]


def measure_per_label(measure, y_true, y_predicted):
    """
    Return per label results of a scikit-learn compatible quality measure

    Parameters
    ----------
    measure : callable
        scikit-compatible quality measure function
    y_true : sparse matrix
        ground truth
    y_predicted : sparse matrix
        the predicted result

    Returns
    -------
    List[int or float]
        score from a given measure depending on what the measure returns
    """
    return [
        measure(y_true[:, i].toarray(), y_predicted[:, i].toarray())
        for i in range(y_true.shape[1])
    ]
