"""
The :mod:`skmultilearn.problem_transform` module provides classifiers 
that follow the problem transformation approaches to multi-label classification:

- :class:`BinaryRelevance` -  treats each label as a separate single-class classification problem 
- :class:`ClassifierChain`-  treats each label as a part of a conditioned chain of single-class classification problems
- :class:`LabelPowerset` - treats each label combination as a separate class with one multi-class classification problem

"""

from .br import BinaryRelevance
from .cc import ClassifierChain
from .lp import LabelPowerset

__all__ = ["BinaryRelevance", 
           "ClassifierChain", 
           "LabelPowerset"]
