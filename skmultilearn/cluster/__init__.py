"""
The :mod:`skmultilearn.cluster` module gathers label space clustering methods.

"""

from .base import LabelSpaceClustererBase
from .base import LabelCooccurenceClustererBase
from .graphtool import GraphToolCooccurenceClusterer
from .igraph import IGraphLabelCooccurenceClusterer
from .matrix import MatrixLabelSpaceClusterer

__all__ = ['LabelSpaceClustererBase',
           'LabelCooccurenceClustererBase',
           'GraphToolCooccurenceClusterer',
           'IGraphLabelCooccurenceClusterer',
           'MatrixLabelSpaceClusterer']
