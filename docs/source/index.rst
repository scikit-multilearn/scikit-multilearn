.. scikit-multilearn documentation master file, created by
   sphinx-quickstart on Thu Nov 27 08:47:28 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scikit-multilearn's documentation!
=============================================

Scikit-multilearn is a BSD-licensed library for multi-label classification that 
is built on top of the well-known `scikit-learn <http://scikit-learn.org>`_ ecosystem. 


How do I start?
---------------

If you are new to multi-label classification alltogether, start with the 
:ref:`Concepts <concepts>` chapter which goes through the concepts and shows where different methods fit.

If you've already performed multi-label classification, you're probably interested in some 
:ref:`examples of how to use scikit-multilearn <classify>` and find out how to: load a data set, learn a classifier and test the results using scikit measures.

If you came here to use the wrapper around the well known `meka library <http://meka.sf.net>`_, there's an example of how to do it in: :ref:`Using the meka wrapper <mekawrapper>`.

If you are a developer and want to join scikit-multilearn, here you can find out how to:

- :ref:`implement a classifier <implementing-classifier>`
- :ref:`implement a new clusterer for ensemble methods based on label space division <implementclusterer>`

.. - :ref:`find some Junior Jobs and help the team <junior-jobs>`


.. todo: write useful scenarios such as CV, train/test split etc.

If you used previous versions of the library, you might want to :ref:`document goes through the changes <changes>`.

Contents:

.. toctree::
   :maxdepth: 2
   
   concepts
   datasets
   classify
   meka
   base
   clusterer
   model_estimation
   api/modules

   changes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

