Meka: python interface to java's meka library
=============================================


Setting up MEKA
---------------
In order to use the python interface to `MEKA <http://meka.sourceforge.net/>`_  you need to have JAVA and MEKA installed. Paths to both are passed to the class's constructor. To set up MEKA download the ``meka-<version>-realease.zip`` from `MEKA's sourceforge page <https://sourceforge.net/projects/meka/>`_, unzip it to ``<MEKA_DIR>`` and use ``<MEKA_DIR>/lib`` as the MEKA classpath for the constructor.

An example path to java might be: ``/usr/bin/java``, an example classpath to meka can be: ``/opt/meka-1.7/lib/``.

**The current version supports meka 1.7**

Using the interface
--------------------
To use the interface class start with importing skmultilearn's module, then create an object of the ``Meka`` class using the constructor and run the interface such as in the example:


.. code-block:: python

	from skmultilearn.dataset import Dataset
	import sklearn.metrics
	import Meka from meka.meka
	

	meka = Meka( 
		meka_classifier = "meka.classifiers.multilabel.LC", 
		weka_classifier = "weka.classifiers.bayes.NaiveBayes",
		meka_classpath = "/opt/meka-1.7/lib/", 
		java_command = '/usr/bin/java')

	predictions, extra_output = meka.run("path_to/train.arff", "path_to/test.arff")

	label_count = int(extra_output['L'])

	# check classifier parameters
	print(extra_output['Classifier_info'])

	# verify results, meka supports big endian only
	input_features, reference_labels = 
		Dataset.load_arff_to_numpy("path_to/test.arff", label_count, endian = "big")

	# measure
	sklearn.metrics.hamming_loss(reference_labels, predictions)

Where:

- ``meka_classifier`` is the MEKA classifier class selected from the `MEKA API <http://meka.sourceforge.net/api-1.7/index.html>`_
- ``weka_classifier`` is the WEKA classifier class selected from the `WEKA API <http://http://weka.sourceforge.net/doc.stable/>`_
