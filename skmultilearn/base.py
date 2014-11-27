import numpy as np

class MethodNotImplementedException(Exception):               pass

class MLClassifierBase(object):
	def __init__(self, classifier = None):
		super(MLClassifierBase, self).__init__()
		self.classifier = classifier

	def clean(self):
		pass

	def generate_data_subset(self, y, labels):
        return [row[labels] for row in y]

	def fit(self, X, y):
		raise MethodNotImplementedException("In MLClassifierBase::fit()")

	def predict(self, X):
		raise MethodNotImplementedException("In MLClassifierBase::fit()")