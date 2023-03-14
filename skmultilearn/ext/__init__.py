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
| :class:`~skmultilearn.ext.Keras`           | Wrapper for the Python Deep Learning library -                   |
|                                            | `KERAS <http://https://keras.io/>`_                              |
+--------------------------------------------+------------------------------------------------------------------+
| :func:`~skmultilearn.ext.download_meka`    | Helper function for installing MEKA                              |
+--------------------------------------------+------------------------------------------------------------------+

"""

import sys, platform
from .meka import Meka, download_meka

__all__ = ["Meka", "download_meka"]

if not (sys.version_info[0] == 2 or platform.architecture()[0] == "32bit"):
    from .keras import Keras

    __all__ += ["Keras"]
