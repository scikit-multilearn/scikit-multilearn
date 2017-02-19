"""
The :mod:`skmultilearn.neurofuzzy` module provides implementations of neural network
and fuzzy approaches to multi-label classification.

Currently the available classes include:

- :class:`MLARAM` - A Multi-Label Hierarchical ARAM Neural Network

"""

from __future__ import absolute_import
from .MLARAMfast import MLARAM


__all__ = ["MLARAM"]
