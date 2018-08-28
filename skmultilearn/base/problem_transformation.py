import numpy as np
from .base import MLClassifierBase
from ..utils import matrix_creation_function_for_format
from scipy.sparse import issparse, csr_matrix


class ProblemTransformationBase(MLClassifierBase):
    """Base class providing common functions for multi-label classifiers
    that follow the problem transformation approach.

    Problem transformation is the approach in which the
    original multi-label classification problem is transformed into one
    or more single-label problems, which are then solved by single-class
    or multi-class classifiers.

    Scikit-multilearn provides a number of such methods:

    - :class:`BinaryRelevance` - performs a single-label single-class classification for each label and sums the results :class:`BinaryRelevance`
    - :class:`ClassifierChains` - performs a single-label single-class classification for each label and sums the results :class:`ClassifierChain`
    - :class:`LabelPowerset` - performs a single-label single-class classification for each label and sums the results :class:`LabelPowerset`

    Parameters
    ----------
    classifier : scikit classifier type
        The base classifier that will be used in a class, will be automagically put under self.classifier for future access.
    require_dense : boolean (default is False)
        Whether the base classifier requires input as dense arrays.
    """

    def __init__(self, classifier=None, require_dense=None):

        super(ProblemTransformationBase, self).__init__()

        self.copyable_attrs = ["classifier", "require_dense"]

        self.classifier = classifier
        if require_dense is not None:
            if isinstance(require_dense, bool):
                self.require_dense = [require_dense, require_dense]
            else:
                assert len(require_dense) == 2 and isinstance(
                    require_dense[0], bool) and isinstance(require_dense[1], bool)
                self.require_dense = require_dense

        else:
            if isinstance(self.classifier, MLClassifierBase):
                self.require_dense = [False, False]
            else:
                self.require_dense = [True, True]

    def _ensure_multi_label_from_single_class(self, matrix, matrix_format='csr'):
        """Transform single class outputs to a 2D sparse matrix
        
        Parameters
        ----------
        matrix : array-like
            input matrix to be checked
        matrix_format : str (default is csr)
            the matrix format to validate with

        Returns
        -------
        scipy.sparse
            a 2-dimensional sparse matrix
        """
        is_2d = None
        dim_1 = None
        dim_2 = None

        # check if array like of array likes
        if isinstance(matrix, (list, tuple, np.ndarray)):
            if isinstance(matrix[0], (list, tuple, np.ndarray)):
                is_2d = True
                dim_1 = len(matrix)
                dim_2 = len(matrix[0])
            # 1d list or array
            else:
                is_2d = False
                # shape is n_samples of 1 class assignment
                dim_1 = len(matrix)
                dim_2 = 1

        # not an array but 2D, probably a matrix
        elif matrix.ndim == 2:
            is_2d = True
            dim_1 = matrix.shape[0]
            dim_2 = matrix.shape[1]

        # what is it? 
        else:
            raise ValueError("Matrix dimensions too large (>2) or other value error")

        new_matrix = None
        if is_2d:
            if issparse(matrix):
                new_matrix = matrix
            else:
                new_matrix = matrix_creation_function_for_format(matrix_format)(matrix, shape=(dim_1, dim_2))
        else:
            new_matrix = matrix_creation_function_for_format(matrix_format)(matrix).T

        assert new_matrix.shape == (dim_1, dim_2)
        return new_matrix
