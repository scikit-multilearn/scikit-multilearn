# Authors: Grzegorz Kulakowski <grzegorz7w@gmail.com>
# License: BSD 3 clause

cimport sklearn.tree._criterion
from sklearn.tree._criterion cimport Criterion

import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef class PCTCiterionBase(Criterion):
    cdef SIZE_t criteria_count #TODO:zmieniec na property
    cdef void proxy_impurity_improvement_all(self, double* all_improvements) nogil
    cdef double pct_hypothesis(self) nogil