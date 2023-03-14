"""
The :mod:`skmultilearn.base` module implements base
classifier classes for scikit-multilearn's multi-label classification.

Two base classifier classes are in use currently in scikit-multilearn:

- :class:`MLClassifierBase` - a generic base class for multi-label classifiers
- :class:`ProblemTransformationBase` - the base class for problem transformation and ensemble approaches that handles a base classifier
"""

from .base import MLClassifierBase
from .problem_transformation import ProblemTransformationBase

__all__ = ["MLClassifierBase", "ProblemTransformationBase"]
