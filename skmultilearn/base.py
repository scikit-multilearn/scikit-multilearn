import numpy as np

class MethodNotImplementedException(Exception):               pass

class MLClassifierBase(object):
	def __init__(self, classifier = None):
		super(MLClassifierBase, self).__init__()
		self.classifier = classifier

	def clean(self):
		pass

	def fit(self, X, y):
		raise MethodNotImplementedException("In MLClassifierBase::fit()")

	def predict(self, X):
		raise MethodNotImplementedException("In MLClassifierBase::fit()")