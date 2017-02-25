.. _loading:

Loading and generating multi-label datasets
===========================================

In this section you will learn how to generate multi-label classification, where to find real world data and how to load it, and how to prepare your data for classification.

The multi-label classification can be performed on different kinds of data - usually it is done on either artificially generated data for analytical purposes or on real-world data sets stored in `ARFF <http://www.cs.waikato.ac.nz/ml/weka/arff.html>` files.

Generating artificial data
--------------------------
Scikit-learn's `sklearn.datasets.make_multilabel_classification <http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html>`_ framework can be used to generate artificial multi-label classification data:

.. code-block:: python

	from sklearn.datasets import make_multilabel_classification

	# this will generate a 
	x, y = make_multilabel_classification(sparse = True, n_labels = 5, 
	  return_indicator = 'sparse', allow_unlabeled = False)


Getting real-world data sets
----------------------------

The ``MULAN`` package provides a `repository <http://mulan.sourceforge.net/datasets-mlc.html>`_ of multi-label datasets used in a variety of publications. The data sets are provided in the ``ARFF`` format, with labels provided as last elements of the ``ARFF`` data frame (little endian). Scikit-multilearn provides support for loading ``ARFF`` data files.

Loading from ARFF
^^^^^^^^^^^^^^^^^

The class :class:`skmultilearn.dataset.Dataset` allows loading data from ``WEKA``, ``MULAN`` or ``MEKA`` provided data sets in ``ARFF`` format. The module depends on `liac-arff <https://pypi.python.org/pypi/liac-arff>`_ and is capable of loading sparse and dense represented ``ARFF`` data, the `X`, `y` are returned as sparse matrices. See the :meth:`skmultilearn.dataset.Dataset.load_arff_to_numpy` for more information. 

Example code for converting ``ARFF`` file to data dumps:

.. code-block:: python

    from skmultilearn.dataset import Dataset
    
    ## some information about the data set 
    # number of labels
    labelcount = 16 
    
    # where the labels are located, 
    # big = at the beginning of the file
    endianness = 'little' 
    
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

The ``scikit-multilearn`` provided data sets are produced using :meth:`skmultilearn.dataset.Dataset` class and contain a dictionary with two keys: ``X``, ``y``, containing a data set in the format described above. The data sets are ``pickle`` dumps compressed using the ``bz2`` module. They can be loaded using the ``Dataset`` class.


Scikit-multilearn data set helpers
----------------------------------
For experimental purposes we provide a helper function to quickly load compressed data sources for multi-label classification. The :meth:`skmultilearn.dataset.Dataset` class can load and save `X` and `y` to a compressed file containing a ``pickle`` of a dictionary with two keys: ``X``, ``y``, containing the input and output matrices. The data sets are dumped using ``pickle`` module and compressed using the ``bz2`` module. They can be loaded using the ``Dataset`` class.

Example use case of  saving and loading data sets:

.. code-block:: python

	from skmultilearn.dataset import Dataset

	Dataset.save_dataset_dump(X, y, "path/filename.dump.bz2")

	X, y = Dataset.load_dataset_dump("path/filename.dump.bz2")


Cross-validation and train-test splits
--------------------------------------

As Tsoumakas et. al `note <http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf>`_: in supervised learning, experiments typically involve a first step of distributing the examples of a dataset into two or more disjoint subsets. When training data abound, the holdout method is used to distribute the examples into a training and a test set, and sometimes also into a validation set. When training data are limited, cross-validation is used, which starts by splitting the dataset into a number of disjoint subsets of approximately equal size.

To perform a train-test split for multi-label classification - having `X` and `y` - we can use scikit-learn's `sklearn.model_selection.train_test_split <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_:

.. code-block:: python

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

	# learn the classifier
	classifier.fit(X_train, y_train)

	# predict labels for test data
	predictions = classifier.predict(X_test)

As there is no established stratified folding procedure for multi-label classification we can use a traditional k-fold cross-validation approach with `sklearn.model_selection.KFold <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html>`_:

.. code-block:: python

    from sklearn.model_selection import KFold

    # remember to set n_splits and shuffle!
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=shuffle)

    for train_index, test_index in kf.split(X, y):
    	# assuming classifier object exists
    	X_train = X[train_index,:]
    	y_train = y[train_index,:]

    	X_test = X[test_index,:]
    	y_test = y[test_index,:]

    	# learn the classifier
    	classifier.fit(X_train, y_train)

    	# predict labels for test data
    	predictions = classifier.predict(X_test)

It is noteworthy that traditional ``k``-folding may lead to severe problems with label combination representability across folds, thus if your data set exhibits a strong label co-occurrence structure you might want to use a label-combination based `stratified k-fold <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_:

.. code-block:: python

    from sklearn.model_selection import StratifiedKFold
    from skmultilearn.problem_transform import LabelPowerset

    lp = LabelPowerset()

    # remember to set n_splits and shuffle!
    kf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=shuffle)

    for train_index, test_index in kf.split(X, lp.transform(y)):

        # assuming classifier object exists
        X_train = X[train_index,:]
        y_train = y[train_index,:]

        X_test = X[test_index,:]
        y_test = y[test_index,:]

        # learn the classifier
        classifier.fit(X_train, y_train)

        # predict labels for test data
        predictions = classifier.predict(X_test)


In the next section you will learn what classification methods are available in scikit-multilearn.

