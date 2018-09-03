License
-------

Scikit-multilearn follows the widely employed copyleft licensing of the scikit
community and is licensed BSD 2-Clause "Simplified" License. You can:

- Use, copy and distribute the unmodified source or binary forms of the licensed program
- Use, copy and distribute modified source or binary forms of the licensed program, provided that all distributed copies are accompanied by the license

.. raw:: html

  <a class="btn" href="https://choosealicense.com/licenses/bsd-2-clause/">Learn mode</a>

GPL library dependency
^^^^^^^^^^^^^^^^^^^^^^

Some of the sub-functionalities of the library uses and depends on GPL-licensed libraries:

- :mod:`skmultilearn.cluster.graphtool` depends on GPL licensed python-graphtool_ module for Stochastic Blockmodel functionality
- :mod:`skmultilearn.cluster.igraph` depends on GPL licensed python-igraph_ module for community detection methods

Using any of these modules incurs GPL on your codebase, thus for commercial purposes for network-based label space division
you should be using the :mod:`skmultilearn.cluster.networkx` module which is depends on the well known BSD-licensed liibrary NetworkX_.

Note that none of these libraries are installed by default.

.. _python-graphtool: https://graph-tool.skewed.de/
.. _python-igraph: http://igraph.org/python/
.. _NetworkX: https://networkx.github.io
