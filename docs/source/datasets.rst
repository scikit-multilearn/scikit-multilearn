.. _datasets:

The data format for multi-label classification
==============================================

In this section you will learn about the data format scikit-multilearn expects.

In order to train a classification model we need data about a phenomenon that the classifier is supposed to generalize. Such data usually comes in two parts:

- the objects that are being classified - the input space - which we will denote as ``X`` and which consists of ``n_samples`` that are represented using ``n_features``
-  the labels assigned to ``n_samples`` objects - an output space - which we will denote as ``y``. ``y`` provides information about which, out of ``n_labels`` that are available, are actually assigned to each of ``n_samples`` objects
 
The multi-label data representation
-----------------------------------

scikit-multilearn expects on input:

- ``X`` to be a matrix of shape ``(n_samples, n_features)``
- ``y`` to be a matrix of shape ``(n_samples, n_labels)`` 

By matrix scikit-multilearn understands following the ``A[i,j]`` element accessing scheme. Sparse matrices are preferred over dense ones. Scikit-multilearn will internally convert dense representations to sparse representations that are most suitable to a given classification procedure. Scikit-multilearn will output 

``X`` can store any type of data a given classification method can handle is allowed, but nominal encoding is always helpful. Nominal encoding is enabled by default when loading data with :meth:`skmultilearn.dataset.Dataset.load_arff_to_numpy` helper, which also returns sparse representations of ``X`` and ``y`` loaded from ARFF data file. 

``y`` is expected to be a binary ``integer`` indicator matrix of shape. In the binary indicator matrix each matrix element ``A[i,j]`` should be either ``1`` if label ``j`` is assigned to a object no ``i``, and ``0`` if not. 

We highly recommend for every multi-label output space to be stored in sparse matrices and expect scikit-multilearn classifiers to operate only on sparse binary label indicator matrices internally. This is also the format of predicted label assignments. Sparse representation is employed as default, because it is very rare for a real world output space ``y`` to be dense. Usually the number of labels assigned per instance is just a small portion of all labels. The average percentage of labels assigned per object is called `label density` and in established data sets it `tends to be really small <http://mulan.sourceforge.net/datasets-mlc.html>`_. 

All matrices should be compatible with the `numpy.matrix <https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html>`_ or `scipy.sparse <https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html>`_ classes, especially in terms of providing two-argument operator to access elements, i.e. the element of ``i``-th row, ``j``-th column of matrix ``A`` should be available via ``A[i,j]`` and not ``A[i][j]``.

Single-label representations in problem transfortion
----------------------------------------------------

The problem transformation approach to multi-label classification converts multi-label problems to single-label problems: single-class or multi-class. Then those problems are solved using base classifiers. Scikit-multilearn maintains compatibility with `scikit-learn data format for single-label classifiers <http://scikit-learn.org/stable/modules/multiclass.html>`_ ,which expect:

- ``X`` to have an `(n_samples, n_features)` shape and be one of the following:

	- an ``array-like`` of ``array-likes``, which usually means a nested array, where ``i``-th row and ``j``-th column are adressed as ``X[i][j]``, in many cases the classifiers expect ``array-like`` to be an ``np.array``
	- a dense matrix of the type ``np.matrix``
	- a scipy sparse matrix  

- ``y`` to be a one-dimensional ``array-like`` of shape ``(n_samples,)`` with one class value per sample, which is a natural representation of a single-label problem


Not all scikit-learn classifiers support matrix input and sparse representations. For this reason every scikit-multilearn classifier that follows a problem transformation approach admits a ``require_dense`` parameter in constructor. As these scikit-multilearn classifiers transform the multi-label problem to a set of single-label problems and solve them using scikit-learn base classifiers - the ``require_dense`` parameter allows control over which format of the transformed input and output space is forwarded to the base classifier.

The parameter ``require_dense`` expects a two-element list: ``[bool or None, bool or None]`` which control the input and output space formats respectively. If None - the base classifier will receive a dense representation if it is does not inherit :class:`skmultilearn.base.MLClassifierBase`, otherwise the representation forwarded will be sparse. The dense representation for ``X`` is a ``numpy.matrix``, while for ``y`` it is a ``numpy.array of int`` (scikit-learn's required format of the output space). 

Scikit-learn's expected format is described `in the scikit-learn docs <http://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format>`_ and assumes that:

- ``X`` is provided either as a ``numpy.matrix``, a ``sparse.matrix`` or as ``array likes of arays likes`` (vectors) of features, i.e. the array of row vectors that consist of input features (same length, i.e. feature/attribute count), ex. a two-object set with each row being a small 1px x 1px image with rgb channels (three ``int8`` values describing red, blue, green colors per pixel): ``[[128,10,10,20,30,128], [10,155,30,10,155,10]]`` - scikit-multilearn will expect a matrix representation and will forward a matrix representation to the base classifier 
- ``y`` is expected to be provided as array of array likes

Some scikit-learn classifiers support sparse representation of ``X`` especially for textual data, to have it forwarded as such to the scikit-learn classifier one needs to pass ``require_dense = [False, None]`` to the scikit-multilearn classifier's constructor. If you are sure that the base classifier you use will be able to handle a sparse matrix representation of ``y`` - pass ``require_dense = [None, False]``. Pass ``require_dense = [False, False]`` if both ``X`` and ``y`` are supported in sparse representation.


Converting from array of arrays
-------------------------------


To convert from scikit-learn 1D representation of array of arrays to matrices, use a relevant scipy matrix/numpy matrix constructor, ex.: 

.. code-block:: python

	import numpy as np
	import scipy.sparse as sp

	# assume X is array of arrays
	np.matrix(X) # yields a dense matrix
	sp.csr_matrix(X) # yields a row-oriented sparse matrix

In the next section we explain how to generate and where to get data for multi-label classification.
