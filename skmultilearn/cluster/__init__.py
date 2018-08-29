"""
The :mod:`skmultilearn.cluster` module gathers label space clustering methods.

"""

from .base import LabelCooccurrenceGraphBuilder
from .fixed import FixedLabelSpaceClusterer
from .matrix import MatrixLabelSpaceClusterer
from .random import RandomLabelSpaceClusterer

__all__ = [
    'FixedLabelSpaceClusterer',
    'LabelCooccurrenceGraphBuilder',
    'MatrixLabelSpaceClusterer',
    'RandomLabelSpaceClusterer'
]

# graphtool import optional (is GPL-ed, does not work on windows)
try:
    from .graphtool import GraphToolLabelGraphClusterer, StochasticBlockModel

    __all__ += ['GraphToolLabelGraphClusterer', 'StochasticBlockModel']
except ImportError:
    pass

# python-igraph import optional (is GPL-ed)
try:
    from .igraph import IGraphLabelGraphClusterer

    __all__ += ['IGraphLabelGraphClusterer']
except ImportError:
    pass

# networkx import is optional
try:
    from .networkx import NetworkXLabelGraphClusterer

    __all__ += ['NetworkXLabelGraphClusterer']
except ImportError:
    pass
