"""
The :mod:`skmultilearn.adapt` module implements algorithm
adaptation approaches to multi-label classification.
"""

from .brknn import BRkNNaClassifier, BRkNNbClassifier
from .mlknn import MLkNN

__all__ = ["BRkNNaClassifier", 
           "BRkNNbClassifier", 
           "MLkNN"]
