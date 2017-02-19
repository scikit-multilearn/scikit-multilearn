"""
The :mod:`skmultilearn.ensemble` module implements ensemble classification schemes
that construct an ensemble of base multi-label classifiers.

Currently the following ensemble classification schemes are available in scikit-multilearn:

- :class:`RakelD` - Distinct RAndom k-labELsets multi-label classifier
- :class:`RakelO` - Overlapping RAndom k-labELsets multi-label classifier.
- :class:`LabelSpacePartitioningClassifier` - a label space partitioning classifier that trains a classifier per label subspace as clustered using methods from :mod:`skmultilearn.cluster`.
- :class:`FixedLabelPartitionClassifier` - a classifier that trains a classifier per label subspace for a given fixed partition

"""

from __future__ import absolute_import
from .rakeld import RakelD
from .rakelo import RakelO
from .fixed import FixedLabelPartitionClassifier, LabelSpacePartitioningClassifier
from .partition import LabelSpacePartitioningClassifier

__all__ = ["RakelD", "RakelO", "LabelSpacePartitioningClassifier",
           "FixedLabelPartitionClassifier", "LabelSpacePartitioningClassifier"]
