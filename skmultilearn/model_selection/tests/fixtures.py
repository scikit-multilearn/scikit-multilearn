import numpy as np

# An example of output space which has enough evidence for 3 folds
# to contain a balanced label and label pair stratification
# but can also illustrate poor stratification behavior
FIXTURE_Y = np.array(
    [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
    ]
)

FOLDS_FIXTURES = [
    # the structure is as follows:
    # (
    #   folds,
    #   example distribution given equal division expectation,
    #   label distribution,
    #   label pair distribution,
    #   no of folds without label evidence,
    #   no of fold-label pairs without evidence,
    #   no of folds without label pairs evidence,
    #   no of fold-labelpair pairs without evidence,
    #   percentage of labels without evidence per fold,
    #   percentage of label pairs without evidence per fold
    # )
    # fold that deliver well-balanced stratification of single-label evidence
    # but fails with label pair evidence in one fold
    (
        [[1, 2, 5], [0, 3, 4]],  # folds
        0.0,  # ED
        0.25,  # LD
        0.25,  # LDP
        0,
        0,
        2,
        3,
        [0.0, 0.0],
        [0.33333333333333337, 0.16666666666666663],
    )
]
