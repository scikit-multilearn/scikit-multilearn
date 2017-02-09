.. _model_estimation:
Estimating parameters
======================

Scikit-multilearn allows estimating parameters to select best models for multi-label classification using scikit-learn's
model selection `GridSearchCV API <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.
In the simplest version it can look for the best parameter of a scikit-multilearn's classifier, which we'll show on the
example case of estimating parameters for MLkNN, and in the more complicated cases of problem transformation methods it
can estimate both the method's hyper parameters and the base classifiers parameter.

Generating data for experimentation
-----------------------------------

Let's start with generating some data

.. code-block:: python

    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split

    x, y = make_multilabel_classification(sparse = True, n_labels = 5,
        return_indicator = 'sparse', allow_unlabeled = False)

Estimating hyper-parameter k for MLkNN
--------------------------------------

In the case of estimating the hyperparameter of a multi-label classifier, we first import the relevant classifier and
scikit-learn's GridSearchCV class. Then we define the values of parameters we want to evaluate. We are interested in which
combination of `k` - the number of neighbours, `s` - the smoothing parameter works best. We also need to select a measure
which we want to optimize - we've chosen the `F1 macro score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_.

After selecting the parameters we intialize and run the cross validation grid search and print the best hyper parameters.

.. code-block:: python

    from skmultilearn.adapt import MLkNN
    from sklearn.model_selection import GridSearchCV

    parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
    score = 'f1-macro

    clf = GridSearchCV(MLkNN(), parameters, scoring=score)
    clf.fit(x, y)

    print clf.best_params_, clf.best_score_

    # output
    ({'k': 1, 's': 0.5}, 0.78988303374297597)

These values can be then used directly with the classifier.

Estimating hyper-parameter k for embedded classifiers
-----------------------------------------------------

In problem transformation classifiers we often need to estimate not only a hyper parameter, but also the parameter of
the base classifier, and also - maybe even the problem transformation method. Let's take a look at this on a three-layer
construction of ensemble of problem transformation classifiers starting with - RAkeLD - the random label space partitioner as the
ensemble classifier, it takes two parameters:
- k: the size of each label partition (k = number of labels)
- classifier: takes a parameter -classifier for transforming multi-label classification problem to a single-label classification one
we will use model search to select which one to use - ClassifierChains or LabelPowerset

Both Classifier Chains and Label Powerset depend on a single-label classifier to perform classification, we will use scikit-learn's
Multinomial Naive Bayes that asks for a smooting parameter ``alpha``. In the following code we wish to find a model that
optimizes parameters of three levels of embedded classifiers.

.. code-block:: python

    from skmultilearn.ensemble.rakeld import RakelD
    from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import MultinomialNB

    x, y = make_multilabel_classification(sparse=True, n_labels=5,
                                          return_indicator='sparse', allow_unlabeled=False)

    parameters = {
        'labelset_size': range(2, 3),
        'classifier': [LabelPowerset(), BinaryRelevance()],
        'classifier__classifier': [MultinomialNB()],
        'classifier__classifier__alpha': [0.7, 1.0],
    }

    clf = GridSearchCV(RakelD(), parameters, scoring='f1_macro')
    clf.fit(x, y)

    print clf.best_params_, clf.best_score_

    # output
    {'labelset_size': 2,
     'classifier__classifier': MultinomialNB(alpha=0.7, class_prior=None, fit_prior=True),
     'classifier': LabelPowerset(classifier=MultinomialNB(alpha=0.7,
      class_prior=None, fit_prior=True),
      require_dense=[True, True]), 'classifier__classifier__alpha': 0.7} 0.690554445761

