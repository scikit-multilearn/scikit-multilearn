# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Grzegorz Kulakowski <grzegorz7w@gmail.com>
#          Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
# License: BSD 3 clause

cimport sklearn.tree._criterion
from sklearn.tree._criterion cimport Criterion #ClassificationCriterion,Gini,
import numpy as np
cimport numpy as np
from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs
from libc.math cimport sqrt
cimport sklearn.tree._utils
from sklearn.tree._utils cimport sizet_ptr_to_ndarray, log
cimport skmultilearn.tree.utils
from skmultilearn.tree.utils cimport safe_realloc,crit_node_impurity,crit_children_impurity


cdef double INFINITY = np.inf


cdef class PCTCiterionBase(Criterion):
    cdef void proxy_impurity_improvement_all(self, double* all_improvements) nogil:
        """Abstract"""
        pass
    cdef double pct_hypothesis(self) nogil:
        """Abstract"""
        return -INFINITY

#ClassificationCriterion from sklearn
cdef class ClassificationCriterion(PCTCiterionBase):
    """Abstract criterion for classification."""

    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs: SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes: numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Count labels for each output
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.n_classes = NULL

        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]

        self.sum_stride = sum_stride

        cdef SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_left = <double*> calloc(n_elements, sizeof(double))
        self.sum_right = <double*> calloc(n_elements, sizeof(double))

        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""

        free(self.n_classes)

    def __reduce__(self):
        return (ClassificationCriterion,
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride,
                   DOUBLE_t* sample_weight, double weighted_n_samples,
                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
        children samples[start:start] and samples[start:end].

        Parameters
        ----------
        y: array-like, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency
        y_stride: SIZE_t
            The stride between elements in the buffer, important if there
            are multiple targets (multi-output)
        sample_weight: array-like, dtype=DTYPE_t
            The weight of each sample
        weighted_n_samples: SIZE_t
            The total weight of all samples
        samples: array-like, dtype=SIZE_t
            A mask on the samples, showing which ones we want to use
        start: SIZE_t
            The first sample to use in the mask
        end: SIZE_t
            The last sample to use in the mask
        """

        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride

        for p in range(start, end):
            i = samples[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0
            if sample_weight != NULL:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <SIZE_t> y[i * y_stride + k]
                sum_total[k * self.sum_stride + c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""

        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_left, 0, n_classes[k] * sizeof(double))
            memcpy(sum_right, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride

    cdef void reverse_reset(self) nogil:
        """Reset the criterion at pos=end."""
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_right, 0, n_classes[k] * sizeof(double))
            memcpy(sum_left, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride

    cdef void update(self, SIZE_t new_pos) nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left child.

        Parameters
        ----------
        new_pos: SIZE_t
            The new ending position for which to move samples from the right
            child to the left child.
        """
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = (k * self.sum_stride +
                                   <SIZE_t> y[i * self.y_stride + k])
                    sum_left[label_index] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = (k * self.sum_stride +
                                   <SIZE_t> y[i * self.y_stride + k])
                    sum_left[label_index] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                sum_right[c] = sum_total[c] - sum_left[c]

            sum_right += self.sum_stride
            sum_left += self.sum_stride
            sum_total += self.sum_stride

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] and save it into dest.

        Parameters
        ----------
        dest: double pointer
            The memory address which we will save the node value into.
        """

        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k

        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride

cdef class GiniBuilder:
    cpdef void add_criterion(self,PCTCriterion pct_criterion):
        pct_criterion.add_criterion(<crit_node_impurity>gini_node_impurity,
                                    <crit_children_impurity>gini_children_impurity)

cdef double gini_node_impurity(ClassificationCriterion crit) nogil:
    cdef SIZE_t* n_classes = crit.n_classes
    cdef double* sum_total = crit.sum_total
    cdef double gini = 0.0
    cdef double sq_count
    cdef double count_k
    cdef SIZE_t k
    cdef SIZE_t c

    for k in range(crit.n_outputs):
        sq_count = 0.0

        for c in range(n_classes[k]):
            count_k = sum_total[c]
            sq_count += count_k * count_k

        gini += 1.0 - sq_count / (crit.weighted_n_node_samples *
                                  crit.weighted_n_node_samples)

        sum_total += crit.sum_stride

    return gini / crit.n_outputs

cdef void gini_children_impurity(ClassificationCriterion crit, double* impurity_left,
                                double* impurity_right) nogil:

    cdef SIZE_t* n_classes = crit.n_classes
    cdef double* sum_left = crit.sum_left
    cdef double* sum_right = crit.sum_right
    cdef double gini_left = 0.0
    cdef double gini_right = 0.0
    cdef double sq_count_left
    cdef double sq_count_right
    cdef double count_k
    cdef SIZE_t k
    cdef SIZE_t c

    for k in range(crit.n_outputs):
        sq_count_left = 0.0
        sq_count_right = 0.0

        for c in range(n_classes[k]):
            count_k = sum_left[c]
            sq_count_left += count_k * count_k

            count_k = sum_right[c]
            sq_count_right += count_k * count_k

        gini_left += 1.0 - sq_count_left / (crit.weighted_n_left *
                                            crit.weighted_n_left)

        gini_right += 1.0 - sq_count_right / (crit.weighted_n_right *
                                              crit.weighted_n_right)

        sum_left += crit.sum_stride
        sum_right += crit.sum_stride

    impurity_left[0] = gini_left / crit.n_outputs
    impurity_right[0] = gini_right / crit.n_outputs

cdef class EntropyBuilder:
    cpdef void add_criterion(self,PCTCriterion pct_criterion):
        pct_criterion.add_criterion(<crit_node_impurity>entropy_node_impurity,
                                    <crit_children_impurity>entropy_children_impurity)

cdef double entropy_node_impurity(ClassificationCriterion crit) nogil:
    """Evaluate the impurity of the current node, i.e. the impurity of
    samples[start:end], using the cross-entropy criterion."""

    cdef SIZE_t* n_classes = crit.n_classes
    cdef double* sum_total = crit.sum_total
    cdef double entropy = 0.0
    cdef double count_k
    cdef SIZE_t k
    cdef SIZE_t c

    for k in range(crit.n_outputs):
        for c in range(n_classes[k]):
            count_k = sum_total[c]
            if count_k > 0.0:
                count_k /= crit.weighted_n_node_samples
                entropy -= count_k * log(count_k)

        sum_total += crit.sum_stride

    return entropy / crit.n_outputs

cdef void entropy_children_impurity(ClassificationCriterion crit, double* impurity_left,
                            double* impurity_right) nogil:
    """Evaluate the impurity in children nodes

    i.e. the impurity of the left child (samples[start:pos]) and the
    impurity the right child (samples[pos:end]).

    Parameters
    ----------
    impurity_left: double pointer
        The memory address to save the impurity of the left node
    impurity_right: double pointer
        The memory address to save the impurity of the right node
    """

    cdef SIZE_t* n_classes = crit.n_classes
    cdef double* sum_left = crit.sum_left
    cdef double* sum_right = crit.sum_right
    cdef double entropy_left = 0.0
    cdef double entropy_right = 0.0
    cdef double count_k
    cdef SIZE_t k
    cdef SIZE_t c

    for k in range(crit.n_outputs):
        for c in range(n_classes[k]):
            count_k = sum_left[c]
            if count_k > 0.0:
                count_k /= crit.weighted_n_left
                entropy_left -= count_k * log(count_k)

            count_k = sum_right[c]
            if count_k > 0.0:
                count_k /= crit.weighted_n_right
                entropy_right -= count_k * log(count_k)

        sum_left += crit.sum_stride
        sum_right += crit.sum_stride

    impurity_left[0] = entropy_left / crit.n_outputs
    impurity_right[0] = entropy_right / crit.n_outputs


cdef class PCTCriterion(ClassificationCriterion):
    cdef SIZE_t max_criteria_count
    cdef crit_node_impurity* criteria_node_impurity
    cdef crit_children_impurity* criteria_children_impurity
    cdef double* avg_all
    cdef double* avg_left
    cdef double* avg_right

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        self.criteria_count = 0
        self.max_criteria_count = 5
        self.criteria_node_impurity = NULL
        self.criteria_children_impurity = NULL
        self.avg_all = NULL
        self.avg_left = NULL
        self.avg_right = NULL
        safe_realloc(&self.criteria_node_impurity, self.max_criteria_count)
        safe_realloc(&self.criteria_children_impurity, self.max_criteria_count)
        safe_realloc(&self.avg_all, self.n_outputs )
        safe_realloc(&self.avg_left,self.n_outputs )
        safe_realloc(&self.avg_right,self.n_outputs )

    def __dealloc__(self):
        """Destructor."""
        free(self.criteria_node_impurity)
        free(self.criteria_children_impurity)
        free(self.avg_all)
        free(self.avg_left)
        free(self.avg_right)


    cpdef void resize_criterion(self, SIZE_t criteria_count):
        safe_realloc(&self.criteria_node_impurity, criteria_count)
        safe_realloc(&self.criteria_children_impurity, criteria_count)
        self.max_criteria_count = criteria_count
        self.criteria_count = min(self.criteria_count, criteria_count)


    #TODO: add resize
    cdef void add_criterion(self, crit_node_impurity criterion_node_impurity, crit_children_impurity criterion_children_impurity) nogil:
        if self.criteria_count >= self.max_criteria_count:
            return #Excetion in use, not visible (nogil restriction)

        self.criteria_node_impurity[self.criteria_count] = criterion_node_impurity
        self.criteria_children_impurity[self.criteria_count] = criterion_children_impurity
        self.criteria_count += 1


    cdef double node_impurity(self) nogil:
        cdef double all_node_impurity = 0
        cdef double actual_node_impurity = -INFINITY
        cdef SIZE_t c = 0

        if self.criteria_count < 1:
            return - INFINITY #Excetion in use, not visible (nogil restriction)
        for c in range(self.criteria_count):
            actual_node_impurity = self.criteria_node_impurity[c](self)
            all_node_impurity += actual_node_impurity / <double>self.criteria_count

        return all_node_impurity


    cdef void children_impurity(self, double* impurity_left,
                            double* impurity_right) nogil:
        """It's backward compatibility implementation of children impurity. All criteria are averaged."""
        cdef SIZE_t c = 0
        cdef double all_impurity_left = 0
        cdef double all_impurity_right = 0

        if self.criteria_count < 1:
            return  #Excetion in use, not visible (nogil restriction)
        for c in range(self.criteria_count):
            self.criteria_children_impurity[c](self, impurity_left, impurity_right)
            all_impurity_left += impurity_left[0] / <double>self.criteria_count
            all_impurity_right += impurity_right[0] / <double>self.criteria_count
        impurity_left[0] = all_impurity_left
        impurity_right[0] = all_impurity_right


    cdef double pct_hypothesis(self) nogil:
        #TODO: add check is multilabel not multiclass
        cdef double var_e
        cdef double var_e_normalized_left
        cdef double var_e_normalized_right
        cdef double E_cout = (self.end-self.start)

        cdef double d_partial_all   =0.0
        cdef double d_partial_left  =0.0
        cdef double d_partial_right =0.0
        cdef double d_all   =0.0
        cdef double d_left  =0.0
        cdef double d_right =0.0

        cdef SIZE_t y_l_label

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t l
        cdef SIZE_t i
        cdef SIZE_t p

        for l in range(self.n_outputs):
            self.avg_all[l] = sum_total[1]/(sum_total[0]+sum_total[1])
            self.avg_left[l] =  sum_left[1]/(sum_left[0]+sum_left[1])
            self.avg_right[l] = sum_right[1]/(sum_right[0]+sum_right[1])

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride

        #TODO: Check situation when pos = end
        for p in range(self.start, self.end):
            i = self.samples[p]
            d_partial_all   =0.0
            d_partial_left  =0.0
            d_partial_right =0.0

            for l in range(self.n_outputs):
                y_l_label = <SIZE_t> self.y[i * self.y_stride + l] #zero or one - it's mutilabel
                d_partial_all += (y_l_label-self.avg_all[l])**2
                if p < self.pos:
                    d_partial_left += (y_l_label-self.avg_left[l])**2
                else:
                    d_partial_right += (y_l_label-self.avg_right[l])**2
            d_all   += sqrt(d_partial_all) #TODO: Maybe (1/E) should be shift to up?
            d_left  += sqrt(d_partial_left)
            d_right += sqrt(d_partial_right)

        var_e = (1.0/E_cout) * d_all
        var_e_normalized_left = (1.0/E_cout)* d_left
        var_e_normalized_right = (1.0/E_cout)* d_right
        h = var_e - (var_e_normalized_left + var_e_normalized_right)
        return h

    cdef void proxy_impurity_improvement_all(self, double* all_improvments) nogil:
        cdef double impurity_left
        cdef double impurity_right
        cdef SIZE_t c = 0
        cdef double actual_proxy_improvement = - INFINITY

        if self.criteria_count < 1:
            return #TODO: Give exception
        for c in range(self.criteria_count):
            self.criteria_children_impurity[c](self, &impurity_left, &impurity_right)
            actual_proxy_improvement = (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)
            all_improvments[c]=actual_proxy_improvement

cdef class PCTAllSklearnCriterions(PCTCriterion):
    def __cinit__(self, SIZE_t n_outputs, np.ndarray[SIZE_t, ndim=1] n_classes):
        gini = GiniBuilder()
        gini.add_criterion(self)
        entropy = EntropyBuilder()
        entropy.add_criterion(self)
        self.criteria_count = 2

    def __reduce__(self):
        return (PCTAllSklearnCriterions,
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())