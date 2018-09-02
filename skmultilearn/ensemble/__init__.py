"""
The :mod:`skmultilearn.ensemble` module implements ensemble classification schemes
that construct an ensemble of base multi-label classifiers.

Currently the following ensemble classification schemes are available in scikit-multilearn:

+--------------------------------------------------------------------+----------------------------------------------------------------+
| Classifier name                                                    | Description                                                    |
+====================================================================+================================================================+
| :class:`~skmultilearn.ensemble.RakelD`                             | Distinct RAndom k-labELsets multi-label classifier             |
+--------------------------------------------------------------------+----------------------------------------------------------------+
| :class:`~skmultilearn.ensemble.RakelO`                             | Overlapping RAndom k-labELsets multi-label classifier.         |
+--------------------------------------------------------------------+----------------------------------------------------------------+
| :class:`~skmultilearn.ensemble.LabelSpacePartitioningClassifier`   | a label space partitioning classifier that trains a            |
|                                                                    | classifier per label subspace as clustered using methods       |
|                                                                    | from :mod:`skmultilearn.cluster`.                              |
+--------------------------------------------------------------------+----------------------------------------------------------------+
| :class:`~skmultilearn.ensemble.MajorityVotingClassifier`           | a label space division classifier that trains a classifier     |
|                                                                    | per label subspace as clustered using methods from             |
|                                                                    | :mod:`skmultilearn.cluster` and assign labels if the majority  |
|                                                                    | of classifiers that contain the label agree on the assignment. |
+--------------------------------------------------------------------+----------------------------------------------------------------+
"""

from __future__ import absolute_import
from .rakeld import RakelD
from .rakelo import RakelO
from .partition import LabelSpacePartitioningClassifier
from .voting import MajorityVotingClassifier

__all__ = ["RakelD", "RakelO", "LabelSpacePartitioningClassifier",
           "LabelSpacePartitioningClassifier", "MajorityVotingClassifier"]
