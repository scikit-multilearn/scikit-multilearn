# scikit-multilearn

[![PyPI version](https://badge.fury.io/py/scikit-multilearn.svg)](https://badge.fury.io/py/scikit-multilearn)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build Status Linux and OSX](https://travis-ci.org/scikit-multilearn/scikit-multilearn.svg?branch=master)](https://travis-ci.org/scikit-multilearn/scikit-multilearn)
[![Build Status Windows](https://ci.appveyor.com/api/projects/status/vd4k18u1lp5btaql/branch/master?svg=true)](https://ci.appveyor.com/project/niedakh/scikit-multilearn/branch/master)

__scikit-multilearn__ is a Python module capable of performing multi-label
learning tasks. It is built on-top of various scientific Python packages
([numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/)) and
follows a similar API to that of [scikit-learn](http://scikit-learn.org/).

- __Website:__ [scikit.ml](http://scikit.ml)
- __Documentation:__ [scikit-multilearn Documentation](http://scikit.ml/api/index.html)


## Features

- __Native Python implementation.__ A native Python implementation for a variety of multi-label classification algorithms. To see the list of all supported classifiers, check this [link](http://scikit.ml/#classifiers).

- __Interface to Meka.__ A Meka wrapper class is implemented for reference purposes and integration. This provides access to all methods available in MEKA, MULAN, and WEKA &mdash; the reference standard in the field.

- __Builds upon giants!__ Team-up with the power of numpy and scikit. You can use scikit-learn's base classifiers as scikit-multilearn's classifiers. In addition, the two packages follow a similar API.

## Dependencies

In most cases you will want to follow the requirements defined in the requirements/*.txt files in the package. 

### Base dependencies
```
scipy
numpy
future
scikit-learn
liac-arff # for loading ARFF files
requests # for dataset module
networkx # for networkX base community detection clusterers
python-louvain # for networkX base community detection clusterers
```

### GPL-incurring dependencies for two clusterers
```
python-igraph # for igraph library based clusterers
python-graphtool # for graphtool base clusterers
```

Note: Installing graphtool is complicated, please see: [graphtool install instructions](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions)

## Installation

To install scikit-multilearn, simply type the following command:

```bash
$ pip install scikit-multilearn
```

This will install the latest release from the Python package index. If you
wish to install the bleeding-edge version, then clone this repository and
run `setup.py`:

```bash
$ git clone https://github.com/scikit-multilearn/scikit-multilearn.git
$ cd scikit-multilearn
$ python setup.py
```

## Basic Usage

Before proceeding to classification,  this library assumes that you have
a dataset with the following matrices:

- `x_train`, `x_test`: training and test feature matrices of size `(n_samples, n_features)`
- `y_train`, `y_test`: training and test label matrices of size `(n_samples, n_labels)`

Suppose we wanted to use a problem-transformation method called Binary
Relevance, which treats each label as a separate single-label classification
problem, to a Support-vector machine (SVM) classifier, we simply perform
the following tasks:

```python
# Import BinaryRelevance from skmultilearn
from skmultilearn.problem_transform import BinaryRelevance

# Import SVC classifier from sklearn
from sklearn.svm import SVC

# Setup the classifier
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False,True])

# Train
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)
```

More examples and use-cases can be seen in the 
[documentation](http://scikit.ml/api/classify.html). For using the MEKA
wrapper, check this [link](http://scikit.ml/api/meka.html#mekawrapper).

## Contributing

This project is open for contributions. Here are some of the ways for
you to contribute:

- Bug reports/fix
- Features requests
- Use-case demonstrations
- Documentation updates

In case you want to implement your own multi-label classifier, please 
read our [Developer's Guide](http://scikit.ml/api/base.html) to help
you integrate your implementation in our API.

To make a contribution, just fork this repository, push the changes
in your fork, open up an issue, and make a Pull Request!

We're also available in Slack! Just go to our [slack group](https://scikit-ml.slack.com/).

## Cite

If you used scikit-multilearn in your research or project, please
cite [our work](https://arxiv.org/abs/1702.01460):

```bibtex
@ARTICLE{2017arXiv170201460S,
   author = {{Szyma{\'n}ski}, P. and {Kajdanowicz}, T.},
   title = "{A scikit-based Python environment for performing multi-label classification}",
   journal = {ArXiv e-prints},
   archivePrefix = "arXiv",
   eprint = {1702.01460},
   year = 2017,
   month = feb
}
```
