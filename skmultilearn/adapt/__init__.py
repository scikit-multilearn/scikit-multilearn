"""
The :mod:`skmultilearn.adapt` module implements algorithm
adaptation approaches to multi-label classification.

Algorithm adaptation methods for multi-label classification 
concentrate on adapting single-label classification algorithms
to the multi-label case usually by changes in cost/decision 
functions. You can find more information about this approach in
the `Mining Multi-label Data  paper 
<http://link.springer.com/chapter/10.1007/978-0-387-09823-4_34>`_ 
by Tsoumakas et. al. 

Currently scikit-multilearn provides implementations of BRkNN and
MLkNN adaptations of the well known k Nearest Neighbours classifier 
adapted by Zhang & Zhou in their paper `A lazy learning approach to 
multi-label learning 
<http://www.sciencedirect.com/science/article/pii/S0031320307000027>`_.

Provides the following classes:

- :class:`BRkNNaClassifier` - a Binary Relevance kNN classifier that assigns a label if at least half of the neighbors are also classified with the label

- :class:`BRkNNbClassifier`-  a Binary Relevance kNN classifier that assigns top m labels of neighbors with m - average number of labels assigned to neighbors

- :class:`MLkNN` - a multi-label adapted kNN classifier with bayesian prior corrections

"""

from .brknn import BRkNNaClassifier, BRkNNbClassifier
from .mlknn import MLkNN

__all__ = ["BRkNNaClassifier", 
           "BRkNNbClassifier", 
           "MLkNN"]
