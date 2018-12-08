"""
The :mod:`skmultilearn.embedding` module provides implementations of label space embedding methods and a general
embedding based classifier.


+--------------------------------------------------------+---------------------------------------------------------------+
| Name                                                   | Description                                                   |
+========================================================+===============================================================+
| :class:`~skmultilearn.embedding.CLEMS`                 | Cost-Sensitive Label Embedding with Multidimensional Scaling  |
+--------------------------------------------------------+---------------------------------------------------------------+
| :class:`~skmultilearn.embedding.OpenNetworkEmbedder`   | Label Network Embedding for Multilabel Classification         |
+--------------------------------------------------------+---------------------------------------------------------------+
| :class:`~skmultilearn.embedding.SKLearnEmbedder`       | Wrapper for scikit-learn embedders                            |
+--------------------------------------------------------+---------------------------------------------------------------+
| :class:`~skmultilearn.embedding.EmbeddingClassifier`   | A general embedding-based classifier                          |
+--------------------------------------------------------+---------------------------------------------------------------+

"""

from .clems import CLEMS
from .openne import OpenNetworkEmbedder
from .sklearn import SKLearnEmbedder
from .classifier import EmbeddingClassifier

__all__ = [
    'CLEMS',
    'OpenNetworkEmbedder',
    'SKLearnEmbedder',
    'EmbeddingClassifier'
]
