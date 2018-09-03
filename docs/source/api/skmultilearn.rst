scikit-multilearn API Reference
===============================

Scikit-multilearn is a BSD-licensed library for multi-label classification that is
built on top of the well-known scikit-learn ecosystem.

Classifiers and tools
---------------------


Algorithm Adaptation approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: skmultilearn.adapt

.. toctree::
    :hidden:

    introduction
    userguide
    authors


Problem Transformation approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: skmultilearn.problem_transform


Ensembles of classifiers
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: skmultilearn.ensemble


Label Space Clusterers
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: skmultilearn.cluster

External classifiers
^^^^^^^^^^^^^^^^^^^^

.. automodule:: skmultilearn.ext


Model selection and data manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: skmultilearn.model_selection

.. automodule:: skmultilearn.dataset


Developer API
-------------

Base classes
^^^^^^^^^^^^

+-------------------------------------------------------------+-------------------------------------------------+
| Name                                                        | Description                                     |
+=============================================================+=================================================+
| :class:`~skmutlilearn.base.MLClassifierBase`                | Base class for multi-label classifiers          |
+-------------------------------------------------------------+-------------------------------------------------+
| :class:`~skmutlilearn.base.ProblemTransformationBase`       | Base class for problem transformation           |
|                                                             | multi-label classifiers                         |
+-------------------------------------------------------------+-------------------------------------------------+
| :class:`~skmutlilearn.cluster.base.GraphBuilderBase`        | Base class for Label Graph builders             |
+-------------------------------------------------------------+-------------------------------------------------+
| :class:`~skmutlilearn.cluster.base.LabelSpaceClustererBase` | Base class for label space clusterers           |
+-------------------------------------------------------------+-------------------------------------------------+
| :class:`~skmutlilearn.cluster.base.LabelGraphClustererBase` | Base class for Label Graph clusterers           |
+-------------------------------------------------------------+-------------------------------------------------+

Modules with helper functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------------+--------------------------------------------------+
| Name                                 | Description                                      |
+======================================+==================================================+
| :mod:`~skmutlilearn.cluster.helpers` | Functions for converting cluster representations |
+--------------------------------------+--------------------------------------------------+
| :mod:`~skmutlilearn.utils`           | Functions for matrix format manipulation         |
+--------------------------------------+--------------------------------------------------+
