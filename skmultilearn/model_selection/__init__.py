"""
The :mod:`skmultilearn.model_selection` module provides implementations multi-label stratification methods
useful for parameter estimation.

Currently the available modules include:

+-------------------------------------------------------------------+-------------------------------------------+
| Name                                                              | Description                               |
+===================================================================+===========================================+
| :class:`~skmultilearn.model_selection.iterative_stratification`   | Iterative stratification                  |
+-------------------------------------------------------------------+-------------------------------------------+
| :mod:`~skmultilearn.model_selection.measures`                     | Stratification quality measures package   |
+-------------------------------------------------------------------+-------------------------------------------+

"""

from .iterative_stratification import (
    IterativeStratification,
    iterative_train_test_split,
)
from .measures import (
    example_distribution,
    label_combination_distribution,
    percentage_of_label_combinations_without_evidence_per_fold,
    folds_without_evidence_for_at_least_one_label_combination,
    folds_label_combination_pairs_without_evidence,
)
