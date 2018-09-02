import random
import unittest

import skmultilearn.dataset as dt

ALL_SET_TEST_CASES = [
    (u'bibtex', u'undivided'),
    (u'bibtex', u'test'),
    (u'bibtex', u'train'),
    (u'birds', u'undivided'),
    (u'birds', u'test'),
    (u'birds', u'train'),
    (u'Corel5k', u'undivided'),
    (u'Corel5k', u'test'),
    (u'Corel5k', u'train'),
    (u'delicious', u'undivided'),
    (u'delicious', u'test'),
    (u'delicious', u'train'),
    (u'emotions', u'undivided'),
    (u'emotions', u'test'),
    (u'emotions', u'train'),
    (u'enron', u'undivided'),
    (u'enron', u'test'),
    (u'enron', u'train'),
    (u'genbase', u'undivided'),
    (u'genbase', u'test'),
    (u'genbase', u'train'),
    (u'mediamill', u'undivided'),
    (u'mediamill', u'test'),
    (u'mediamill', u'train'),
    (u'medical', u'undivided'),
    (u'medical', u'test'),
    (u'medical', u'train'),
    (u'rcv1subset1', u'undivided'),
    (u'rcv1subset1', u'test'),
    (u'rcv1subset1', u'train'),
    (u'rcv1subset2', u'undivided'),
    (u'rcv1subset2', u'test'),
    (u'rcv1subset2', u'train'),
    (u'rcv1subset3', u'undivided'),
    (u'rcv1subset3', u'test'),
    (u'rcv1subset3', u'train'),
    (u'rcv1subset4', u'undivided'),
    (u'rcv1subset4', u'test'),
    (u'rcv1subset4', u'train'),
    (u'rcv1subset5', u'undivided'),
    (u'rcv1subset5', u'test'),
    (u'rcv1subset5', u'train'),
    (u'scene', u'undivided'),
    (u'scene', u'test'),
    (u'scene', u'train'),
    (u'tmc2007_500', u'undivided'),
    (u'tmc2007_500', u'test'),
    (u'tmc2007_500', u'train'),
    (u'yeast', u'undivided'),
    (u'yeast', u'test'),
    (u'yeast', u'train')
]

# to save time don't download all possible sets always, only check before release
NUMBER_OF_SETS_TO_DOWNLOAD_IN_TEST = 2
# NUMBER_OF_SETS_TO_DOWNLOAD_IN_TEST = len(ALL_SET_TEST_CASES)

class ClassifierBaseTest(unittest.TestCase):
    def test_all_data_is_present(self):
        for data_set, variant in dt.available_data_sets():
            self.assertIn((data_set, variant), ALL_SET_TEST_CASES)

    def test_dataset_works(self):
        data_home = dt.get_data_home(data_home=None, subdirectory='test')
        for set_name, variant in random.sample(ALL_SET_TEST_CASES, NUMBER_OF_SETS_TO_DOWNLOAD_IN_TEST):
            X, y, feature_names, label_names = dt.load_dataset(set_name=set_name, variant=variant , data_home=data_home)
            self.assertEqual(len(X.shape), 2)
            self.assertEqual(len(y.shape), 2)
            self.assertEqual(len(feature_names), X.shape[1])
            self.assertEqual(len(label_names), y.shape[1])
            self.assertEqual(X.shape[0], y.shape[0])

        dt.clear_data_home(data_home)


