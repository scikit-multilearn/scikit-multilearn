import os
import shlex
import subprocess
import sys
import tempfile
import zipfile
from builtins import filter
from builtins import map
from builtins import range
from builtins import str

import scipy.sparse as sparse

from ..base import MLClassifierBase
from ..dataset import save_to_arff, get_data_home, _download_single_file, _get_md5

try:
    from shlex import quote as cmd_quote
except ImportError:
    from pipes import quote as cmd_quote

SUPPORTED_VERSION = '1.9.2'
SUPPORTED_VERSION_MD5 = 'e909044b39513bbad451b8d71098b22c'


def download_meka(version=None):
    """Downloads a given version of the MEKA library and returns its classpath

    Parameters
    ----------
    version : str
        the MEKA version to download, default falls back to currently supported version 1.9.2

    Returns
    -------
    string
        meka class path string for installed version

    Raises
    ------
    IOError
        if unpacking the meka release file does not provide a proper setup
    Exception
        if MD5 mismatch happens after a download error
    """
    version = version or SUPPORTED_VERSION
    meka_release_string = "meka-release-{}".format(version)
    file_name = meka_release_string + '-bin.zip'
    meka_path = get_data_home(subdirectory='meka')
    target_path = os.path.join(meka_path, file_name)
    path_to_lib = os.path.join(meka_path, meka_release_string, 'lib')

    if os.path.exists(target_path):
        print("MEKA {} found, not downloading".format(version))

    else:
        print("MEKA {} not found, downloading".format(version))
        release_url = "http://downloads.sourceforge.net/project/meka/meka-{}/".format(version)
        _download_single_file(file_name, target_path, release_url)

        found_md5 = _get_md5(target_path)
        if SUPPORTED_VERSION_MD5 != found_md5:
            raise Exception("MD5 mismatch - possible MEKA download error")

    if not os.path.exists(path_to_lib):
        with zipfile.ZipFile(target_path, 'r') as meka_zip:
            print("Unzipping MEKA {} to {}".format(version, meka_path + os.path.sep))
            meka_zip.extractall(path=meka_path + os.path.sep)

    if not os.path.exists(os.path.join(path_to_lib, 'meka-{}.jar'.format(version))):
        raise IOError("Something went wrong, MEKA files missing, please file a bug report")

    return path_to_lib + os.path.sep


class Meka(MLClassifierBase):
    """Wrapper for the MEKA classifier

    Allows using MEKA, WEKA and some of MULAN classifiers from scikit-compatible API. For more information on
    how to use this class see the tutorial: :doc:`../meka`

    Parameters
    ----------
    meka_classifier : str
        The MEKA classifier string and parameters from the MEKA API,
        such as :code:`meka.classifiers.multilabel.MULAN -S RAkEL2`
    weka_classifier : str
        The WEKA classifier string and parameters from the WEKA API,
        such as :code:`weka.classifiers.trees.J48`
    java_command : str
        Path to test the java command
    meka_classpath: str
        Path to the MEKA class path folder, usually the folder lib
        in the directory MEKA was extracted into

    Attributes
    ----------
    output_ : str
        the full text output of MEKA command

    References
    ----------

    If you use this wrapper please also cite:

    .. code-block :: latex

        @article{MEKA,
            author = {Read, Jesse and Reutemann, Peter and Pfahringer, Bernhard and Holmes, Geoff},
            title = {{MEKA}: A Multi-label/Multi-target Extension to {Weka}},
            journal = {Journal of Machine Learning Research},
            year = {2016},
            volume = {17},
            number = {21},
            pages = {1--5},
            url = {http://jmlr.org/papers/v17/12-164.html},
        }

        @article{Hall:2009:WDM:1656274.1656278,
            author = {Hall, Mark and Frank, Eibe and Holmes, Geoffrey and Pfahringer, Bernhard and Reutemann, Peter and Witten, Ian H.},
            title = {The WEKA Data Mining Software: An Update},
            journal = {SIGKDD Explor. Newsl.},
            issue_date = {June 2009},
            volume = {11},
            number = {1},
            month = nov,
            year = {2009},
            issn = {1931-0145},
            pages = {10--18},
            numpages = {9},
            url = {http://doi.acm.org/10.1145/1656274.1656278},
            doi = {10.1145/1656274.1656278},
            acmid = {1656278},
            publisher = {ACM},
            address = {New York, NY, USA},
        }

    Examples
    --------

    Here's an example of performing Label Powerset classification using MEKA with a WEKA Naive Bayes classifier.

    .. code-block:: python

        from skmultilearn.ext import Meka, download_meka

        meka = Meka(
            meka_classifier = "meka.classifiers.multilabel.LC",
            weka_classifier = "weka.classifiers.bayes.NaiveBayes",
            meka_classpath = download_meka(),
            java_command = '/usr/bin/java')

        meka.fit(X_train, y_train)
        predictions = meka.predict(X_test)

    """

    def __init__(self, meka_classifier=None, weka_classifier=None,
                 java_command=None, meka_classpath=None):
        super(Meka, self).__init__()

        self.java_command = java_command
        if self.java_command is None:
            # TODO: this will not be needed once we're python 3 ready - we will
            # use it only in python 2.7 cases
            from whichcraft import which
            self.java_command = which("java")

            if self.java_command is None:
                raise ValueError("Java not found")

        self.meka_classpath = meka_classpath
        if self.meka_classpath is None:
            self.meka_classpath = os.environ.get('MEKA_CLASSPATH')

            if self.meka_classpath is None:
                raise ValueError("No meka classpath defined")

        self.meka_classifier = meka_classifier
        self.weka_classifier = weka_classifier

        self.copyable_attrs = [
            'meka_classifier',
            'weka_classifier',
            'java_command',
            'meka_classpath'
        ]
        self.output_ = None
        self._verbosity = 5
        self._warnings = None
        self.require_dense = [False, False]

        self._clean()

    def _clean(self):
        """Sets various attributes to :code:`None`"""
        self._results = None
        self._statistics = None
        self.output_ = None
        self._error = None
        self._label_count = None
        self._instance_count = None

    def _remove_temporary_files(self, temporary_files):
        """Internal function for cleaning temporary files"""
        for file_object in temporary_files:
            file_name = file_object.name
            file_object.close()
            if os.path.exists(file_name):
                os.remove(file_name)

            arff_file_name = file_name + '.arff'
            if os.path.exists(arff_file_name):
                os.remove(arff_file_name)

    def fit(self, X, y):
        """Fits classifier to training data

        Internally this method dumps X and y to temporary arff files and
        runs MEKA with relevant arguments using :meth:`_run`. It uses a
        sparse DOK representation (:class:`scipy.sparse.dok_matrix`)
        of the X matrix.

        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments

        Returns
        -------
        self
            fitted instance of self
        """
        self._clean()
        X = self._ensure_input_format(
            X, sparse_format='dok', enforce_sparse=True)
        y = self._ensure_output_format(
            y, sparse_format='dok', enforce_sparse=True)
        self._label_count = y.shape[1]

        # we need this in case threshold needs to be recalibrated in meka
        self.train_data_ = save_to_arff(X, y)
        train_arff = tempfile.NamedTemporaryFile(delete=False)
        classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)

        try:
            with open(train_arff.name + '.arff', 'w') as fp:
                fp.write(self.train_data_)

            input_args = [
                '-verbosity', "0",
                '-split-percentage', "100",
                '-t', '"{}"'.format(train_arff.name + '.arff'),
                '-d', '"{}"'.format(classifier_dump_file.name),
            ]

            self._run_meka_command(input_args)
            self.classifier_dump = None
            with open(classifier_dump_file.name, 'rb') as fp:
                self.classifier_dump = fp.read()
        finally:
            self._remove_temporary_files([train_arff, classifier_dump_file])

        return self

    def predict(self, X):
        """Predict label assignments for X

        Internally this method dumps X to temporary arff files and
        runs MEKA with relevant arguments using :func:`_run`. It uses a
        sparse DOK representation (:class:`scipy.sparse.dok_matrix`)
        of the X matrix.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            sparse matrix of integers with shape :code:`(n_samples, n_features)`
        """
        X = self._ensure_input_format(
            X, sparse_format='dok', enforce_sparse=True)
        self._instance_count = X.shape[0]

        if self.classifier_dump is None:
            raise Exception('Not classified')

        sparse_y = sparse.coo_matrix((X.shape[0], self._label_count), dtype=int)

        try:
            train_arff = tempfile.NamedTemporaryFile(delete=False)
            test_arff = tempfile.NamedTemporaryFile(delete=False)
            classifier_dump_file = tempfile.NamedTemporaryFile(delete=False)

            with open(train_arff.name + '.arff', 'w') as fp:
                fp.write(self.train_data_)

            with open(classifier_dump_file.name, 'wb') as fp:
                fp.write(self.classifier_dump)

            with open(test_arff.name + '.arff', 'w') as fp:
                fp.write(save_to_arff(X, sparse_y))

            args = [
                '-l', '"{}"'.format(classifier_dump_file.name)
            ]

            self._run(train_arff.name + '.arff', test_arff.name + '.arff', args)
            self._parse_output()

        finally:
            self._remove_temporary_files(
                [train_arff, test_arff, classifier_dump_file]
            )

        return self._results

    def _run(self, train_file, test_file, additional_arguments=[]):
        """Runs the meka classifiers

        Parameters
        ----------
        train_file : str
            path to train :code:`.arff` file in meka format
            (big endian, labels first in attributes list).
        test_file : str
            path to test :code:`.arff` file in meka format
            (big endian, labels first in attributes list).

        Returns
        -------
        predictions: sparse binary indicator matrix [n_test_samples, n_labels]
            array of binary label vectors including label predictions of
            shape :code:`(n_test_samples, n_labels)`
        """
        self.output_ = None
        self._warnings = None

        # meka_command_string = 'java -cp "/home/niedakh/pwr/old/meka-1.5/lib/*" meka.classifiers.multilabel.MULAN -S RAkEL2
        # -threshold 0 -t {train} -T {test} -verbosity {verbosity} -W weka.classifiers.bayes.NaiveBayes'
        # meka.classifiers.multilabel.LC, weka.classifiers.bayes.NaiveBayes

        args = [
                   '-t', '"{}"'.format(train_file),
                   '-T', '"{}"'.format(test_file),
                   '-verbosity', str(5),
               ] + additional_arguments

        self._run_meka_command(args)
        return self

    def _parse_output(self):
        """Internal function for parsing MEKA output."""
        if self.output_ is None:
            self._results = None
            self._statistics = None
            return None

        predictions_split_head = '==== PREDICTIONS'
        predictions_split_foot = '|==========='

        if self._label_count is None:
            self._label_count = map(lambda y: int(y.split(')')[1].strip()), [
                x for x in self.output_.split('\n') if 'Number of labels' in x])[0]

        if self._instance_count is None:
            self._instance_count = int(float(filter(lambda x: '==== PREDICTIONS (N=' in x, self.output_.split(
                '\n'))[0].split('(')[1].split('=')[1].split(')')[0]))
        predictions = self.output_.split(predictions_split_head)[1].split(
            predictions_split_foot)[0].split('\n')[1:-1]

        predictions = [y.split(']')[0]
                             for y in [x.split('] [')[1] for x in predictions]]
        predictions = [[a for a in [f.strip() for f in z.split(',')] if len(a) > 0]
                             for z in predictions]
        predictions = [[int(a) for a in z] for z in predictions]

        assert self._verbosity == 5

        self._results = sparse.lil_matrix(
            (self._instance_count, self._label_count), dtype='int')
        for row in range(self._instance_count):
            for label in predictions[row]:
                self._results[row, label] = 1

        statistics = [x for x in self.output_.split(
            '== Evaluation Info')[1].split('\n') if len(x) > 0 and '==' not in x]
        statistics = [y for y in [z.strip() for z in statistics] if '  ' in y]
        array_data = [z for z in statistics if '[' in z]
        non_array_data = [z for z in statistics if '[' not in z]

        self._statistics = {}
        for row in non_array_data:
            r = row.strip().split('  ')
            r = [z for z in r if len(z) > 0]
            r = [z.strip() for z in r]
            if len(r) < 2:
                continue
            try:
                test_value = float(r[1])
            except ValueError:
                test_value = r[1]

            r[1] = test_value
            self._statistics[r[0]] = r[1]

        for row in array_data:
            r = row.strip().split('[')
            r = [z.strip() for z in r]
            r[1] = r[1].replace(', ', ' ').replace(
                ',', '.').replace(']', '').split(' ')
            r[1] = [x for x in r[1] if len(x) > 0]
            self._statistics[r[0]] = r[1]

    def _run_meka_command(self, args):
        command_args = [
            self.java_command,
            '-cp', '"{}*"'.format(self.meka_classpath),
            self.meka_classifier,
        ]

        if self.weka_classifier is not None:
            command_args += ['-W', self.weka_classifier]

        command_args += args

        meka_command = " ".join(command_args)

        if sys.platform != 'win32':
            meka_command = shlex.split(meka_command)

        pipes = subprocess.Popen(meka_command,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        self.output_, self._error = pipes.communicate()
        if type(self.output_) == bytes:
            self.output_ = self.output_.decode(sys.stdout.encoding)
        if type(self._error) == bytes:
            self._error = self._error.decode(sys.stdout.encoding)

        if pipes.returncode != 0:
            raise Exception(self.output_ + self._error)
