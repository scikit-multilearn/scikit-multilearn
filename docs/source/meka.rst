.. _mekawrapper:
Using the meka wrapper
======================

Setting up MEKA
---------------
In order to use the python interface to `MEKA <http://meka.sourceforge.net/>`_  you need to have JAVA and MEKA installed. Paths to both are passed to the class's constructor. To set up MEKA download the ``meka-<version>-realease.zip`` from `MEKA's sourceforge page <https://sourceforge.net/projects/meka/>`_, unzip it to ``<MEKA_DIR>`` and use ``<MEKA_DIR>/lib`` as the MEKA classpath for the constructor.

An example path to java might be: ``/usr/bin/java``, an example classpath to meka can be: ``/opt/meka-1.9/lib/``.

**The current version supports meka 1.9**

Note that you will need to have ``liac-arff`` installed if you want to use the MEKA wrapper, you can get them using: ``pip install liac-arff``.


Using the wrapper
--------------------
Starting from scikit-multilearn ``0.0.2`` the meka wrapper is available from ``skmultilearn.ext`` (ext as in external) and is a fully scikit-compatible multi-label classifier.

To use the interface class start with importing skmultilearn's module, then create an object of the ``Meka`` class using the constructor and run the interface such as in the example:


.. code-block:: python

    from sklearn.datasets import make_multilabel_classification
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import hamming_loss
    from skmultilearn.ext import Meka


    X, y = make_multilabel_classification(sparse = True, 
        return_indicator = 'sparse')

    X_train, X_test, y_train, y_test = train_test_split(X, 
        y, 
        test_size=0.33)

    meka = Meka( 
        meka_classifier = "meka.classifiers.multilabel.LC", 
        weka_classifier = "weka.classifiers.bayes.NaiveBayes",
        meka_classpath = "/opt/meka-1.9/lib/", 
        java_command = '/usr/bin/java')

    meka.fit(X_train, y_train)

    predictions = meka.predict(X_test)

    metrics.hamming_loss(y_test, predictions)

Where:

- ``meka_classifier`` is the MEKA classifier class selected from the `MEKA API <http://meka.sourceforge.net/api-1.7/index.html>`_
- ``weka_classifier`` is the WEKA classifier class selected from the `WEKA API <http://http://weka.sourceforge.net/doc.stable/>`_
- ``java_command`` is the path to java, if not provided, the wrapper will try to import the `whichcraft <https://pypi.python.org/pypi/whichcraft>`_ module to find java in the system PATH
- meka_classpath is the path to where meka.jar and weka.jar are located, usually they come together in meka releases, so this points to the ``lib`` subfolder of the folder where ``meka-<version>-realease.zip`` file was unzipped. If not provided the path is taken from environmental variable: ``MEKA_CLASSPATH``

A good introduction to selecting MEKA or WEKA classifiers can be found on `MEKA's Methods page <http://meka.sourceforge.net/methods.html>`_.


Remember that if you use MEKA, apart from citing scikit-multilearn, you should also cite both MEKA and WEKA papers:

.. code-block:: latex

    @article{MEKA,
        author = {Read, Jesse and Reutemann, Peter and Pfahringer, Bernhard and Holmes, Geoff},
        title = {{MEKA}: A Multi-label/Multi-target Extension to {Weka}},
        journal = {Journal of Machine Learning Research},
        year = {2016},
        volume = {17},
        number = {21},
        pages = {1--5},
        url = {http://jmlr.org/papers/v17/12-164.html},
    }

    @article{Hall:2009:WDM:1656274.1656278,
        author = {Hall, Mark and Frank, Eibe and Holmes, Geoffrey and Pfahringer, Bernhard and Reutemann, Peter and Witten, Ian H.},
        title = {The WEKA Data Mining Software: An Update},
        journal = {SIGKDD Explor. Newsl.},
        issue_date = {June 2009},
        volume = {11},
        number = {1},
        month = nov,
        year = {2009},
        issn = {1931-0145},
        pages = {10--18},
        numpages = {9},
        url = {http://doi.acm.org/10.1145/1656274.1656278},
        doi = {10.1145/1656274.1656278},
        acmid = {1656278},
        publisher = {ACM},
        address = {New York, NY, USA},
    } 

