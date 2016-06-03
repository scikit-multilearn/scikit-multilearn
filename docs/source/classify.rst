.. _classify:
Classify your dataset
=====================

To classify data with a multi-label classifier, you need to have:

- selected a base classifier, ex. a naive bayes one
- selected the multi-label classification method, ex. Binary Relevance
- an array or (dense/sparse) matrix with feature vectors to classify
- a training data set: 
    - ``X``: array of input feature vectors
    - ``y`` arary of binary label indicator vectors associated with each row

Loading the data set
--------------------
``scikit-multilearn`` lets you load two formats of data, one of them is the traditional ARFF format, using `liac-arff <https://pythonhosted.org/liac-arff/>`_ package. The other uses scikit-multilearn format that is a BZ2-pped pickle of a dict containing two sparse matrices.

Loading from arff
^^^^^^^^^^^^^^^^^

This approach uses the ``liac-arff`` library to load ARFF files into ``scikit-multilearn``. Many benchmark ARFF data sets can be found in `MULAN's collection <http://mulan.sourceforge.net/datasets-mlc.html>`_. This can be done using the :class:`Dataset` static method :func:`Dataset::load_arff_to_numpy`.

.. code-block:: python

    from skmultilearn.dataset import Dataset
    
    ## some information about the data set 
    # number of labels
    labelcount = 16 
    
    # where the labels are located, 
    # big = at the beginning of the file
    endianness = 'big' 
    
    # dtype used in the feature space
    feature_type = 'float' 
    
    # whether the nominal attributes should be encoded as integers
    encode_nominal = True

    # if True - use the sparse loading mechanism from liac-arff
    # if False - load dense representation and convert to sparse
    load_sparse = True

    # load data
    X_train, y_train = Dataset.load_arff_to_numpy("path_to_data/dataset-train.dump.bz2", 
        labelcount = labelcount, 
        endian = "big", 
        input_feature_type = feature_type,
        encode_nominal = encode_nominal,
        load_sparse = load_sparse)

    X_test, y_test = Dataset.load_arff_to_numpy("path_to_data/dataset-train.dump.bz2",
        labelcount = labelcount, 
        endian = "big", 
        input_feature_type = feature_type,
        encode_nominal = encode_nominal,
        load_sparse = load_sparse)


Classifying input using scikit-multilearn
-----------------------------------------

The easiest way to perform multi-label classification is to transform it to a single-label classification task and then use scikit classifiers for that new job. There is a variety of such classifiers available in the :module:`skmultilearn.problem_transformation` module. :class:`Binary Relevance` is an example of such approach. 

Binary Relevance requires a base classifier to use in the single-label problem. It clones a new one for each label and performs per label classification, summing the results together in the end. It can be done using scikit-multilearn as follows.

.. code-block:: python

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.naive_bayes import GaussianNB
    import sklearn.metrics

    # assume data is loaded using 
    # and is available in X_train/X_test, y_train/y_test

    # initialize Binary Relevance multi-label classifier 
    # with gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB(), require_dense)
    
    # train
    classifier.fit(X_train, y_train)
    
    # predict
    predictions = classifier.predict(X_test)

    # measure
    print(sklearn.metrics.hamming_loss(y_test, predictions))


As described in :class:`ProblemTransformationBase`, the ``requires_dense`` parameter can be used to make scikit-multilearn pass sparse representations of data down to scikit-learn (only a few classifiers support this). While ``scikit-multilearn`` uses sparse matrices everywhere, ``scikit-learn`` is still in transition - to enable this (and a large speed up) use the following example:

.. code-block:: python

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.svm import SVC
    import sklearn.metrics

    # assume data is loaded using 
    # and is available in X_train/X_test, y_train/y_test

    # initialize Binary Relevance multi-label classifier 
    # with an SVM classifier
    # SVM in scikit only supports the X matrix in sparse representation 
    classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])

    # train
    classifier.fit(X_train, y_train)
    
    # predict
    predictions = classifier.predict(X_test)

    # measure
    print(sklearn.metrics.hamming_loss(y_test, predictions))
