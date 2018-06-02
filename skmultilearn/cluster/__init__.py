"""
The :mod:`skmultilearn.cluster` module gathers label space clustering methods.
"""

from .base import LabelSpaceNetworkClustererBase, LabelCooccurenceGraphBuilder, GraphBuilderBase
from .matrix import MatrixLabelSpaceClusterer
from .random import RandomLabelSpaceClusterer

__all__ = ['LabelSpaceNetworkClustererBase',
           'GraphBuilderBase',
           'LabelCooccurenceGraphBuilder',
           'MatrixLabelSpaceClusterer',
           'RandomLabelSpaceClusterer']

# graphtool import optional (is GPL-ed, does not work on windows)
try:
    from .graphtool import GraphToolCooccurenceClusterer
    __all__ += ['GraphToolCooccurenceClusterer']
except ImportError:
    pass

# python-igraph import optional (is GPL-ed)
try:
    from .igraph import IGraphLabelCooccurenceClusterer

    __all__ += ['IGraphLabelCooccurenceClusterer']
except ImportError:
    pass


# networkx import is optional
try:
    from .networkx import NetworkXLabelCooccurenceClusterer
    __all__ += ['NetworkXLabelCooccurenceClusterer']
except ImportError:
pass