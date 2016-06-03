.. _changes:
Changes from previous releases
==============================

To implement a multi-label classifier you need to subclass :meth:`skmultilearn.base.MLClassifierBase`. The constructor takes the underlying base classifier and sets it to ``self.classifier`` for future deep copying and access. In order to implement a new classifier you should implement the :meth:`skmultilearn.base.MLClassifierBase.fit` and :meth:`skmultilearn.base.MLClassifierBase.predict` methods.

0.0.1 -> 0.0.3
--------------
The ``0.0.2`` release has been removed due to error in upload to PyPi and the idiotic decision of the PyPi admins to disallow updating files.

Sparse representation
^^^^^^^^^^^^^^^^^^^^^^

In ``0.0.2`` the entire ``scikit-multilearn`` package has been converted to use sparse representation of input and output spaces internally. Helper functions were introduced for this purpose in the :module:`utils` module and problem transformation base class. Multi-label output spaces are usually very sparse. This change provides a large speed up in problem transformation and ensemble approaches. In problem transformation classifiers a new parameter - ``requires_dense`` of type ``[bool, bool]`` which denotes whether the base classifier requires dense input for ``X`` - first ``bool`` - or ``y`` - second ``bool``. If no values passed, the base class infers ``[False, False]`` for :class:`skmultilearn.base.MLClassifierBase` derived classes and ``[True, True]``.

Meka is scikit-compatible
^^^^^^^^^^^^^^^^^^^^^^^^^

The meka classifier has been updated to support version ``1.9`` and is now a fully ``scikit-learn`` compatible classifier supporting ``fit`` and ``predict`` among others. 
