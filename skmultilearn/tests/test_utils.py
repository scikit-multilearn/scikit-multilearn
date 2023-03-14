import unittest
import numpy as np
import scipy.sparse as sp

from ..utils import get_matrix_in_format, matrix_creation_function_for_format

SPARSE_MATRIX_FORMATS = ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]


class UtilsTest(unittest.TestCase):
    def test_if_get_matrix_ensures_type(self):
        matrix = sp.csr_matrix([])
        for sparse_format in SPARSE_MATRIX_FORMATS:
            new_matrix = get_matrix_in_format(matrix, sparse_format)

            self.assertTrue(sp.issparse(new_matrix))
            self.assertTrue(new_matrix.format == sparse_format)

    def test_if_matrix_creation_follows_format(self):
        matrix = np.matrix([])
        for sparse_format in SPARSE_MATRIX_FORMATS:
            new_matrix = matrix_creation_function_for_format(sparse_format)(matrix)

            self.assertTrue(sp.issparse(new_matrix))
            self.assertTrue(new_matrix.format == sparse_format)

    def test_ensure_get_matrix_does_not_clone_if_format_agrees(self):
        matrix = np.matrix([])
        for sparse_format in SPARSE_MATRIX_FORMATS:
            created_matrix = matrix_creation_function_for_format(sparse_format)(matrix)
            converted_matrix = get_matrix_in_format(created_matrix, sparse_format)

            self.assertTrue(id(created_matrix) == id(converted_matrix))


if __name__ == "__main__":
    unittest.main()
