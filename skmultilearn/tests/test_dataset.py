import random
import unittest

import skmultilearn.dataset as dt

ALL_SET_TEST_CASES = [
    ("bibtex", "undivided"),
    ("bibtex", "test"),
    ("bibtex", "train"),
    ("birds", "undivided"),
    ("birds", "test"),
    ("birds", "train"),
    ("Corel5k", "undivided"),
    ("Corel5k", "test"),
    ("Corel5k", "train"),
    ("delicious", "undivided"),
    ("delicious", "test"),
    ("delicious", "train"),
    ("emotions", "undivided"),
    ("emotions", "test"),
    ("emotions", "train"),
    ("enron", "undivided"),
    ("enron", "test"),
    ("enron", "train"),
    ("genbase", "undivided"),
    ("genbase", "test"),
    ("genbase", "train"),
    ("mediamill", "undivided"),
    ("mediamill", "test"),
    ("mediamill", "train"),
    ("medical", "undivided"),
    ("medical", "test"),
    ("medical", "train"),
    ("rcv1subset1", "undivided"),
    ("rcv1subset1", "test"),
    ("rcv1subset1", "train"),
    ("rcv1subset2", "undivided"),
    ("rcv1subset2", "test"),
    ("rcv1subset2", "train"),
    ("rcv1subset3", "undivided"),
    ("rcv1subset3", "test"),
    ("rcv1subset3", "train"),
    ("rcv1subset4", "undivided"),
    ("rcv1subset4", "test"),
    ("rcv1subset4", "train"),
    ("rcv1subset5", "undivided"),
    ("rcv1subset5", "test"),
    ("rcv1subset5", "train"),
    ("scene", "undivided"),
    ("scene", "test"),
    ("scene", "train"),
    ("tmc2007_500", "undivided"),
    ("tmc2007_500", "test"),
    ("tmc2007_500", "train"),
    ("yeast", "undivided"),
    ("yeast", "test"),
    ("yeast", "train"),
]

# to save time don't download all possible sets always, only check before release
NUMBER_OF_SETS_TO_DOWNLOAD_IN_TEST = 2
# NUMBER_OF_SETS_TO_DOWNLOAD_IN_TEST = len(ALL_SET_TEST_CASES)


class ClassifierBaseTest(unittest.TestCase):
    def test_all_data_is_present(self):
        for data_set, variant in dt.available_data_sets():
            self.assertIn((data_set, variant), ALL_SET_TEST_CASES)

    def test_dataset_works(self):
        data_home = dt.get_data_home(data_home=None, subdirectory="test")
        for set_name, variant in random.sample(
            ALL_SET_TEST_CASES, NUMBER_OF_SETS_TO_DOWNLOAD_IN_TEST
        ):
            X, y, feature_names, label_names = dt.load_dataset(
                set_name=set_name, variant=variant, data_home=data_home
            )
            self.assertEqual(len(X.shape), 2)
            self.assertEqual(len(y.shape), 2)
            self.assertEqual(len(feature_names), X.shape[1])
            self.assertEqual(len(label_names), y.shape[1])
            self.assertEqual(X.shape[0], y.shape[0])

        dt.clear_data_home(data_home)
