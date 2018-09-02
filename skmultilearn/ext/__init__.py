"""
The :mod:`skmultilearn.ext` provides wrappers for other multi-label
classification libraries. Currently it provides a wrapper for:

Currently the available classes include:

+--------------------------------------------+------------------------------------------------------------------+
| Name                                       | Description                                                      |
+============================================+==================================================================+
| :class:`~skmultilearn.ext.Meka`            | Wrapper for the Multilabel Extension to WEKA -                   |
|                                            | `MEKA <http://meka.sf.net>`_ library                             |
+--------------------------------------------+------------------------------------------------------------------+
| :func:`~skmultilearn.ext.download_meka`    | Helper function for installing MEKA                              |
+--------------------------------------------+------------------------------------------------------------------+

"""

from .meka import Meka, download_meka

__all__ = ["Meka", 'download_meka']
