Welcome to scikit-multilearn's documentation!
=============================================

Scikit-multilearn is a BSD-licensed library for multi-label classification that 
is built on top of the well-known `scikit-learn <http://scikit-learn.org>`_ ecosystem. 


How do I start?
---------------

If you are new to multi-label classification alltogether, start with the 
:ref:`Concepts <concepts>` chapter which goes through the concepts and shows where different methods fit.

If you've already performed multi-label classification or you've used scikit-learn before, you should:

1. read about :ref:`datasets` in scikit-multilearn
2. learn about :ref:`loading`
3. start the fun with classification by :ref:`classify`
4. see how you can improve your results by :ref:`model_estimation`

If you came here to use the wrapper around the well known `meka library <http://meka.sf.net>`_, there's an example of how to do this in: :ref:`mekawrapper`.

If you are a developer and want to join scikit-multilearn, here you can find out how to:

- :ref:`implement a classifier <implementing-classifier>`
- :ref:`implement a new clusterer for ensemble methods based on label space division <implementclusterer>`

Some candidates for implementation are listed in the `Junior Jobs section of the website <http://scikit.ml/#tasks>`_ if you'd like to help.

.. toctree::
   :caption: User Guide

   concepts
   datasets
   loading
   classify

.. toctree::
   :caption: Tutorials

   model_estimation
   meka


.. toctree::
   :caption: Developer's Guide

   base
   clusterer
   Changelog <https://github.com/scikit-multilearn/scikit-multilearn/blob/master/CHANGES.md>
   API documentation <api/skmultilearn>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

