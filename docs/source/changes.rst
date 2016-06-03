.. _changes:
Changes from previous releases
==============================

To implement a multi-label classifier you need to subclass :meth:`skmultilearn.base.MLClassifierBase`. The constructor takes the underlying base classifier and sets it to ``self.classifier`` for future deep copying and access. In order to implement a new classifier you should implement the :meth:`skmultilearn.base.MLClassifierBase.fit` and :meth:`skmultilearn.base.MLClassifierBase.predict` methods.

0.0.1 -> 0.0.2
--------------

Sparse representation
^^^^^^^^^^^^^^^^^^^^^^

The ``fit(self, X, y)`` expects classifier training data represented as an array-like of input feature vectors (rows) ``X`` and an array-like of binary label vectors as described in :ref:`datasets`. It should return ``self`` after the classifier has been fitted to training data.

Meka is scikit-compatible
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``predict(self ,X)`` expects an array-like of input feature vectors (rows) ``X`` that are to be classified. It should return an array-like of binary label vectors as described in :ref:`datasets`.


Problem transformations changed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``predict(self ,X)`` expects an array-like of input feature vectors (rows) ``X`` that are to be classified. It should return an array-like of binary label vectors as described in :ref:`datasets`.

