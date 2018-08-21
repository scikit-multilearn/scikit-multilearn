"""
The :mod:`skmultilearn.ext` provides wrappers for other multi-label
classification libraries. Currently it provides a wrapper for:

- :class:`Meka` - the Multilabel Extension to WEKA - `MEKA <http://meka.sf.net>`_ library and through it to `MULAN <http://mulan.sf.net>`_ by Tsoumakas. et. al.

"""

from .meka import Meka, download_meka

__all__ = ["Meka", 'download_meka']
