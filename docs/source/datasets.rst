.. _datasets:

The data format for multi-label classification
==============================================

In this section you will learn about the data format scikit-multilearn expects.

In order to train a classification model, i.e. a classifier, we need data about a phenomenon that the classifier is supposed to generalize, this data usually comes in two parts:

- data about the objects that are being classified, an input space, which we will denote as `X`
- data about an output space which we will denote as `y`


What does scikit-multilearn expect?
-----------------------------------

In a given multi-label classification problem it is very rare for the output space matrix of label assignments  ``y`` to be dense. Usually the number of labels assigned per instance is just a small portion of all labels. The average percentage of labels assigned per instance is called `label density` and in established data sets it `tends to be really small for most data sets <http://mulan.sourceforge.net/datasets-mlc.html>`_. 

For this reason by default scikit-multilearn expects the ``X`` and ``y`` data to be provided as sparse matrices. Please note that while dense matrices are supported as well, they will be converted to sparse matrices for internal use in scikit-multilearn classifiers and only sparse matrices will be returned.

The output matrix ``y`` is expected to be a binary indicator matrix, thus if ``X`` has a shape of ``n_samples`` with ``n_features``, then ``y`` should have the shape of ``n_samples`` rows and ``n_labels`` columns, each column with values ``1`` if label ``j`` is assigned to a given row and ``0`` if not.

The sparse matrices should be compatible with the `numpy.matrix <https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html>`_ or `scipy.sparse <https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html>`_ classes, especially in terms of providing two-argument operator to access elements, i.e. the element of ``i``-th row, ``j``-th column of matrix ``A`` should be available via ``A[i,j]`` and not ``A[i][j]``.

If a non-scikit multilearn base classifiers if used (from scikit-learn, tensorflow or similar), for example in problem transformation, it should support matrix arguments, as described above API. A dense matrix representation of ``X`` and ``y`` will be passed to non scikit-multilearn base classifier's ``fit(X,y)`` and ``predict(X)`` methods. A ``require_dense`` argument is provided in case a base classifier accepts a sparse matrix representation for ``X`` or ``y``. 

If you pass ``require_dense = [False, False]`` to a constructor of a class which takes a ``classifier`` argument, it will cause the ``X`` and ``y`` to be passed as sparse matrices to the base classifier cloned `classifier` objects. In this two-element array of booleans the first one is responsible for the representation of ``X`` and the second for ``y``.

Converting from array of arrays
-------------------------------

Scikit-learn and numpy sometimes expect an array of array's representation, ex. in case of three labels and two samples, ``y`` would look as follows: ``[[0, 1, 1], [1,0 ,1]]``. The expected format is described `in the scikit-learn docs <http://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format>`_ and assumes that:

- ``X`` is provided as array of arays (vectors) of features, i.e. the array of row vectors that consist of input features (same length, i.e. feature/attribute count), ex. a two-object set with each row being a small 1px x 1px image with rgb channels (three ``int8`` values describing red, blue, green colors per pixel): ``[[128,10,10,20,30,128], [10,155,30,10,155,10]]`` - scikit-multilearn will expect a matrix representation 
- ``y`` in case of multi-label problems is provided as binary indicator array of arrays

To convert from scikit-learn 1D representation of array of arrays to matrices, use a relevant scipy matrix/numpy matrix constructor, ex.: 

.. code-block:: python

	import numpy as np
	import scipy.sparse as sp

	# assume X is array of arrays
	np.matrix(X) # yields a dense matrix
	sp.csr_matrix(X) # yields a row-oriented sparse matrix

Note. that scikit-learn estimators should support 2D matrix representations, yet they may return array of array representations.

In the next section we explain how to generate and where to get data for multi-label classification.