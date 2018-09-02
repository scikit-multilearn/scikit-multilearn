"""
The :mod:`skmultilearn.problem_transform` module provides classifiers
that follow the problem transformation approaches to multi-label classification.

The problem transformation approach to multi-label classification converts multi-label problems to
single-label problems: single-class or multi-class.


+----------------------------------------------------------+------------------------------------------------+
| Classifier                                               | Description                                    |
+==========================================================+================================================+
| :class:`~skmultilearn.problem_transform.BinaryRelevance` |  treats each label as a separate single-class  |
|                                                          |  classification problem                        |
+----------------------------------------------------------+------------------------------------------------+
| :class:`~skmultilearn.problem_transform.ClassifierChain` |  treats each label as a part of a conditioned  |
|                                                          |  chain of single-class classification problems |
+----------------------------------------------------------+------------------------------------------------+
| :class:`~skmultilearn.problem_transform.LabelPowerset`   | treats each label combination as a separate    |
|                                                          | class with one multi-class classification      |
|                                                          | problem                                        |
+----------------------------------------------------------+------------------------------------------------+


"""

from .br import BinaryRelevance
from .cc import ClassifierChain
from .lp import LabelPowerset

__all__ = ["BinaryRelevance", 
           "ClassifierChain", 
           "LabelPowerset"]
