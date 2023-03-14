"""
The :mod:`skmultilearn.adapt` module implements algorithm
adaptation approaches to multi-label classification.

Algorithm adaptation methods for multi-label classification 
concentrate on adapting single-label classification algorithms
to the multi-label case usually by changes in cost/decision 
functions.

Currently the following algorithm adaptation classification schemes are available in scikit-multilearn:

+-----------------------------------------------+-----------------------------------------------------------+
| Classifier                                    | Description                                               |
+===============================================+===========================================================+
| :class:`~skmultilearn.adapt.BRkNNaClassifier` | a Binary Relevance kNN classifier that assigns a label    |
|                                               | if at least half of the neighbors are also classified     |
|                                               | with the label                                            |
+-----------------------------------------------+-----------------------------------------------------------+
| :class:`~skmultilearn.adapt.BRkNNbClassifier` | a Binary Relevance kNN classifier that assigns top m      |
|                                               | labels of neighbors with m - average number of labels     |
|                                               | assigned to neighbors                                     |
+-----------------------------------------------+-----------------------------------------------------------+
| :class:`~skmultilearn.adapt.MLkNN`            | a multi-label adapted kNN classifier with bayesian prior  |
|                                               | corrections                                               |
+-----------------------------------------------+-----------------------------------------------------------+
| :class:`~skmultilearn.adapt.MLARAM`           | a multi-Label Hierarchical ARAM Neural Network            |
+-----------------------------------------------+-----------------------------------------------------------+
| :class:`~skmultilearn.adapt.MLTSVM`           | twin multi-Label Support Vector Machines                  |
+-----------------------------------------------+-----------------------------------------------------------+

"""

from .brknn import BRkNNaClassifier, BRkNNbClassifier
from .mlknn import MLkNN
from .mlaram import MLARAM
from .mltsvm import MLTSVM

__all__ = ["BRkNNaClassifier", "BRkNNbClassifier", "MLkNN", "MLARAM", "MLTSVM"]
