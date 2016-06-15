import numpy as np
import scipy.sparse as sp

SPARSE_FORMAT_TO_CONSTRUCTOR = {
    "bsr": sp.bsr_matrix,
    "coo": sp.coo_matrix,
    "csc": sp.csc_matrix,
    "csr": sp.csr_matrix,
    "dia": sp.dia_matrix,
    "dok": sp.dok_matrix,
    "lil": sp.lil_matrix
}

def get_matrix_in_format(original_matrix, matrix_format):
    if isinstance(original_matrix, np.ndarray):
        return SPARSE_FORMAT_TO_CONSTRUCTOR[matrix_format](original_matrix)

    if original_matrix.getformat() == matrix_format:
        return original_matrix

    return original_matrix.asformat(matrix_format)


def matrix_creation_function_for_format(sparse_format):
    if sparse_format not in SPARSE_FORMAT_TO_CONSTRUCTOR:
        return None

    return SPARSE_FORMAT_TO_CONSTRUCTOR[sparse_format]
