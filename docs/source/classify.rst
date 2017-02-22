.. _classify:

Selecting a multi-label classifier
==================================

In this document you will learn:

- what classifier approaches are available in scikit-multilearn
- how to perform classification

This section assumes that you have prepared a data set for classification and:

- ``x_train``, ``x_test`` variables contain input feature train and test matrices
- ``y_train``, ``y_test`` variables contain output label train and test matrices


As we noted in :ref:`concepts` multi-label classification can be performed under three approaches:

- algorithm adaption approach 
- problem transformation approach
- ensemble of multi-label classifiers approach


Adapted algorithms
------------------

The algorithm adaptation approach is based on a single-label classification method adapted for multi-label classification problem. Scikit-multilearn provides:

- k-NearestNeighbours classifiers adapted to multi-label purposes by `Zhang et, al. <http://www.sciencedirect.com/science/article/pii/S0031320307000027>`_: ``BrkNN``, ``MLkNN`` available from :mod:`skmultilearn.adapt`
- a hierarchical neurofuzzy classifier HARAM adapted to multi-label purposes by `Benites et. al. <https://kops.uni-konstanz.de/handle/123456789/33471>`_ ``ML-ARAM`` available in :mod:`skmultilearn.neurofuzzy`.

Algorithm adaption methods methods usually require parameter estimation.  Selecting best parameters of algorithm adaptation classifiers is discussed in :ref:`model_estimation`.

An example code for using :class:`skmultilearn.adapt.MLkNN` looks like this:

.. code-block:: python

    from skmultilearn.adapt import MLkNN

    classifier = MLkNN(k=3)
    
    # train
    classifier.fit(X_train, y_train)
    
    # predict
    predictions = classifier.predict(X_test)


Problem transformation
----------------------

Problem transformation approaches are provided in the :mod:`skmultilearn.problem_transform` module and they require a selection of a scikit-learn compatible single-label base classificatier that will be cloned one or more times during the problem transformation. Scikit-learn provides a variety of base classifiers such as:

- `decision trees <http://scikit-learn.org/stable/modules/tree.html>`_
- `Support Vector Machines <http://scikit-learn.org/stable/modules/svm.html>`_
- `Stochastic Gradient Descent <http://scikit-learn.org/stable/modules/sgd.html>`_
- `Nearest Neighbors <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
- `Naive Bayesian Classifiers <http://scikit-learn.org/stable/modules/naive_bayes.html>`_ 

Scikit-multilearn provides three problem transformation approaches:

- :class:`skmultilearn.problem_transform.BinaryRelevance` -  treats each label as a separate single-class classification problem 
- :class:`skmultilearn.problem_transform.ClassifierChain`-  treats each label as a part of a conditioned chain of single-class classification problems
- :class:`skmultilearn.problem_transform.LabelPowerset` - treats each label combination as a separate class with one multi-class classification problem

Problem transformation classifiers take two arguments:

- ``classifier`` - an instance of a base classifier object, to be cloned and refitted upon the multi-label classifiers ``fit`` stage
- ``require_dense`` - a ``[boolean, boolean]`` governing whether the base classifier receives dense or sparse arguments. It is explained in detail in :ref:`datasets`


An example of a Label Powerset transformation from multi-label classification to a single-label multi-class problem to be solved using a Gaussian Naive Bayes classifier:

.. code-block:: python

    from skmultilearn.problem_transform import LabelPowerset
    from sklearn.naive_bayes import GaussianNB

    # initialize Label Powerset multi-label classifier 
    # with a gaussian naive bayes base classifier
    classifier = LabelPowerset(GaussianNB())
    
    # train
    classifier.fit(X_train, y_train)
    
    # predict
    predictions = classifier.predict(X_test)

By default the base classifier will be provided with a dense representation, but some scikit-learn classifiers also support sparse representations. This is an example use of a Binary Relevance classifier with a single-class SVM classifier that does can handle sparse input matrix:

.. code-block:: python

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.svm import SVC

    # initialize Binary Relevance multi-label classifier 
    # with an SVM classifier
    # SVM in scikit only supports the X matrix in sparse representation

    classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])

    # train
    classifier.fit(X_train, y_train)
    
    # predict
    predictions = classifier.predict(X_test)



Ensemble approaches
-------------------

It is often useful to train more than one model for a subset of labels in multi-label classification, especially for large label spaces - a well-selected smaller label subspace `can allow more efficient classification <http://www.mdpi.com/1099-4300/18/8/282>`_. For this purpose the module implements ensemble classification schemes that construct an ensemble of base multi-label classifiers.

Currently the following ensemble classification schemes are available in scikit-multilearn:

- :class:`skmultilearn.ensemble.RakelD` - Distinct RAndom k-labELsets multi-label classifier
- :class:`skmultilearn.ensemble.RakelO` - Overlapping RAndom k-labELsets multi-label classifier.
- :class:`skmultilearn.ensemble.LabelSpacePartitioningClassifier` - a label space partitioning classifier that trains a classifier per label subspace as clustered using methods from :mod:`skmultilearn.cluster`.
- :class:`skmultilearn.ensemble.FixedLabelPartitionClassifier` - a classifier that trains a classifier per label subspace for a given fixed partition

An example code for an ensemble of RandomForests under a Label Powerset multi-label classifiers trained for each label subspace - partitioned using fast greedy community detection methods on a label co-occurrence graph looks like this:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from skmultilearn.problem_transform import LabelPowerset
    from skmultilearn.cluster import IGraphLabelCooccurenceClusterer
    from skmultilearn.ensemble import LabelSpacePartitioningClassifier

    # construct base forest classifier
    base_classifier = RandomForestClassifier()

    # setup problem transformation approach with sparse matrices for random forest
    problem_transform_classifier = LabelPowerset(classifier=base_classifier, 
        require_dense=[False, False])

    # partition the label space using fastgreedy community detection
    # on a weighted label co-occurrence graph with self-loops allowed
    clusterer = IGraphLabelCooccurenceClusterer('fastgreedy', weighted=True, 
        include_self_edges=True)

    # setup the ensemble metaclassifier
    classifier = LabelSpacePartitioningClassifier(problem_transform_classifier, clusterer)

    # train
    classifier.fit(X_train, y_train)
    
    # predict
    predictions = classifier.predict(X_test)


MEKA classifiers
----------------

In a situation when one needs a method not yet implemented in scikit-multilearn - a MEKA/MULAN wrapper is provided and described in section :ref:`mekawrapper`.
