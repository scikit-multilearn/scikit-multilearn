.. _concepts:

Concepts guide
==============

In this section you will learn the basic concepts behind multi-label classification.

Aim
---

Classification aims to assign classes/labels to objects. Objects usually represent things we come across in daily life: photos, audio recordings, text documents, videos, but can also include complicated biological systems. 

Objects are usually represented by their selected features (its count denoted as ``n_features`` in the documentation). Features are the characteristics of objects that distinguish them from others. For example text documents can be represented by words that are present in them. 

The output of classification for a given object is either a class or a set of classes. Traditional classification, usually due to computational limits, aimed at solving only single-label scenarios in which at most one class had been assigned to an object.

Single-label vs multi-label classification
------------------------------------------

One can identify two types of single-label classification problems:

- a single-class one, where the decision is whether to assign the class or not, for ex. having a photo sample from someones pancreas, deciding if it is a photo of cancer sample or not. This is also sometimes called binary classification, as the output values of the predictions are always ``0`` or ``1``
- a multi-class problem where the class, if assigned, is selected from a number of available classes: for example, assigning a brand to a photo of a car

In multi-label classification one can assign more than one label/class out of the available `n_labels` to a given object.

`Madjarov et al. <http://kt.ijs.si/DragiKocev/wikipage/lib/exe/fetch.php?media=2012pr_ml_comparison.pdf>`_ divide approaches to multi-label classification into three categories, you should select a scikit-multilearn base class according to the philosophy behind your classifier: 

- algorithm adaptation, currently none in ``scikit-multilearn`` in the future they will be placed in ``skmultilearn.adapt``
- problem transformation, such as Binary Relevance, Label Powerset & more, are now available from ``skmultilearn.problem_transformation``
- ensemble classification, such as ``RAkEL`` or label space partitioning classifiers, are now available from ``skmultilearn.ensemble``

A single-label classifier is a function that given an object represented as a feature vector of length ``n_features`` assigns a class (a number, or None). A multi-label classifier outputs a set of assigned labels, either in a form of a list of assigned labels or as a binary vector in which a ``1`` or ``0`` on ``i``-th position indicates if an ``i``-th label is assigned or not.

To learn a classifier we use a training set that provides ``n_samples`` of sampled objects represented by ``n_features`` with evidence concerning which labels out of ``n_labels`` are assigned to each of the object. The quality of the classifier is tested on a test set that follows the same format.

In the next section you will learn about the data format scikit-multilearn expects from the training and test sets.