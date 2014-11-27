Classifying input using scikit-multilearn
-----------------------------------------

To classify data with a multi-label classifier, you need to have:

- a training data set: ``X`` - array of input feature vectors and ``y`` arary of binary label indicator vectors associated with each row

- selected a base classifier, ex. a naive bayes one

- selected the multi-label classification method, ex. Binary Relevance

- an array of input feature vectors you want to have labeled

An example use case of the data sets for classification:

.. code-block:: python

	from skmultilearn.dataset import Dataset
	from skmultilearn.meta.br import BinaryRelevance
	from sklearn.naive_bayes import GaussianNB
	import sklearn.metrics

	# load data
	train_set = Dataset.load_dataset_dump("data/scene-train.dump.bz2")
	test_set = Dataset.load_dataset_dump("data/scene-test.dump.bz2")

	# initialize Binary Relevance multi-label classifier with gaussian naive bayes base classifier
	classifier = BinaryRelevance(GaussianNB())
	
	# train
	classifier.fit(train_set['X'],train_set['y'])
	
	# predict
	predictions = classifier.predict(test_set['X'])

	# measure
	print(sklearn.metrics.hamming_loss(test_set['y'], predictions))
