from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import filter
from builtins import str
from builtins import range
from builtins import object
import arff
import bz2
import pickle
import numpy as np
import os
import csv
import sys
import shutil
import urllib.request
import urllib.parse
import urllib.error
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os.path import splitext
from os import listdir
from os import makedirs
from scipy import sparse
import hashlib


def get_data_home(data_home=None):
    """Return the path of the scikit-multilearn data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'scikit_ml_learn_data'
    in the user home folder.

    Alternatively, it can be set by the 'SCIKIT_ML_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    :param data_home string or None: the path to the directory in which scikit-multilearn
        data sets should be stored, if None the path is generated as stated above

    :returns: the path to the data home
    :rtype: string

    """
    if data_home is None:
        data_home = environ.get('SCIKIT_ML_LEARN_DATA',
                                join('~', 'scikit_ml_learn_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    :param data_home string or None: the path to the directory in which scikit-multilearn
        data sets should be stored

    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def get_download_base_url():
    """Returns base URL for data sets."""

    return 'http://scikit.ml/datasets/'


def get_dataset_list():
    """Loads data set list

    The format of the list is a follows:

    - each row corresponds to a variant of a data set
    - variants include: train, test and undivided, note that sometimes data
        sets are not provided in train, test division by their authors
    - in each row column 0 is the md5, column 1 is the file name available
        under :func:`get_download_base_url`

    """

    f = urllib.request.urlopen(get_download_base_url() + "data.list")
    raw_data_list = f.read()
    return raw_data_list


def available_data_sets():
    """Lists available data sets and their variants

    :returns: list of available data sets and their variants
    :rtype dict[set_name] with list of variants:

    """

    archives = get_dataset_list()
    archives = [x.split(';')[-1].split('.')[0].split('-')
                for x in archives.split('\n')]
    variants = {}
    for a in archives:
        if a[0] not in variants:
            variants[a[0]] = []
        if len(a) > 1:
            variants[a[0]].append(a[-1])
    return variants


def download_dataset(set_name, variant):
    """Downloads a data set

    :param set_name string: name of set from :func:`available_data_sets`
    :param variant string: variant of the data set

    :returns: path to the downloaded data set file
    """

    def get_md5(file_name):
        hash_md5 = hashlib.md5()
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    data_sets = get_dataset_list()
    data_sets = [x.split(';') for x in data_sets.split('\n')]

    filter_function = lambda x: variant in x[1] and set_name in x[1]

    for md5, name in filter(filter_function, data_sets):
        target_name = join(get_data_home(), name)
        if exists(target_name):
            if md5 == get_md5(target_name):
                print (
                    "{} - exists, not redownloading".format(set_name, variant))

                return target_name

            else:
                print (
                    "{} - exists, but MD5 sum mismatch - redownloading".format(set_name, variant))
        else:
            print("{} - does not exists downloading".format(set_name, variant))

        # not found or broken md5
        urllib.request.urlretrieve(get_download_base_url() + name, target_name)
        found_md5 = get_md5(target_name)
        if md5 != found_md5:
            raise Exception(
                "{}: MD5 mismatch {} vs {} - possible download error".format(name, md5, found_md5))

        print("Downloaded {}-{}".format(set_name, variant))

        return target_name

    return None


def load_dataset(set_name, variant):
    """Loads a selected variant of the given data set

    :param set_name string: name of set from :func:`available_data_sets`
    :param variant string: variant of the data set

    :returns: the loaded multilabel data set variant in the scikit-multilearn format, see data_sets

    """

    path = download_dataset(set_name, variant)
    if path is not None:
        return load_dataset_dump(path)

    return None


def load_from_arff(filename, labelcount, endian="big", input_feature_type='float', encode_nominal=True, load_sparse=False, return_attribute_definitions=False):
    """Method for loading ARFF files as numpy array

    Parameters
    ----------

    filename : string
        Path to ARFF file

    labelcount: integer
        Number of labels in the ARFF file

    endian: string{"big", "little"}
        Whether the ARFF file contains labels at the beginning of the attributes list ("big" endianness, MEKA format)
        or at the end ("little" endianness, MULAN format)

    input_feature_type: numpy.type as string
        The desire type of the contents of the return 'X' array-likes, default 'i8',
        should be a numpy type, see http://docs.scipy.org/doc/numpy/user/basics.types.html

    encode_nominal: boolean
        Whether convert categorical data into numeric factors - required for some scikit classifiers that can't handle non-numeric input featuers.

    load_sparse: boolean
        Whether to read arff file as a sparse file format, liac-arff breaks if sparse reading is enabled for non-sparse ARFFs.

    Returns
    -------

    X: scipy sparse matrix with ``input_feature_type`` elements,
    y: scipy sparse matrix of binary label indicator matrix

    """
    matrix = None
    if not load_sparse:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.DENSE)
        matrix = sparse.csr_matrix(
            arff_frame['data'], dtype=input_feature_type)
    else:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.COO)
        data = arff_frame['data'][0]
        row = arff_frame['data'][1]
        col = arff_frame['data'][2]
        matrix = sparse.coo_matrix(
            (data, (row, col)), shape=(max(row) + 1, max(col) + 1))

    X, y = None, None

    if endian == "big":
        X, y = matrix.tocsc()[:, labelcount:].tolil(), matrix.tocsc()[
            :, :labelcount].astype(int).tolil()
    elif endian == "little":
        X, y = matrix.tocsc()[
            :, :-labelcount].tolil(), matrix.tocsc()[:, -labelcount:].astype(int).tolil()
    else:
        # unknown endian
        return None

    if return_attribute_definitions:
        return X, y, arff_frame['attributes']
    else:
        return X, y


def save_to_arff(X, y, endian="little", save_sparse=True):
    """Method for dumping data to ARFF files

    Parameters
    ----------

    filename : string
        Path to ARFF file

    labelcount: integer
        Number of labels in the ARFF file

    endian: string{"big", "little"}
        Whether the ARFF file contains labels at the beginning of the attributes list ("big" endianness, MEKA format)
        or at the end ("little" endianness, MULAN format)

    save_sparse: boolean
        Whether to read arff file as a sparse file format, liac-arff breaks if sparse reading is enabled for non-sparse ARFFs.

    Returns
    -------

    string: the ARFF dump string

    """
    X = X.todok()
    y = y.todok()

    x_prefix = 0
    y_prefix = 0

    x_attributes = [(u'X{}'.format(i), u'NUMERIC')
                    for i in range(X.shape[1])]
    y_attributes = [(u'y{}'.format(i), [str(0), str(1)])
                    for i in range(y.shape[1])]

    if endian == "big":
        y_prefix = X.shape[1]
        relation_sign = -1
        attributes = x_attributes + y_attributes

    elif endian == "little":
        x_prefix = y.shape[1]
        relation_sign = 1
        attributes = y_attributes + x_attributes

    else:
        raise ValueError("Endian not in {big, little}")

    if save_sparse:
        data = [{} for r in range(X.shape[0])]
    else:
        data = [[0 for c in range(X.shape[1] + y.shape[1])]
                for r in range(X.shape[0])]

    for keys, value in list(X.items()):
        data[keys[0]][x_prefix + keys[1]] = value

    for keys, value in list(y.items()):
        data[keys[0]][y_prefix + keys[1]] = value

    dataset = {
        u'description': u'traindata',
        u'relation': u'traindata: -C {}'.format(y.shape[1] * relation_sign),
        u'attributes': attributes,
        u'data': data
    }

    return arff.dumps(dataset)


def save_dataset_dump(filename, input_space, labels, feature_names, label_names):
    """Saves a compressed data set dump

    Parameters
    ----------

    filename : string
        Path to dump file, if without .bz2, the .bz2 extension will be appended.

    input_space: array-like of array-likes
        Input space array-like of input feature vectors

    labels: array-like of binary label vectors
        Array-like of labels assigned to each input vector, as a binary indicator vector (i.e. if 5th position has value 1
        then the input vector has label no. 5)

    feature_names: array-like
        optional, names of features

    label_names: array-like
        optional, names of labels
    """
    if filename[-4:] != '.bz2':
        filename += ".bz2"

    with bz2.BZ2File(filename, "wb") as file_handle:
        pickle.dump(
            {'X': input_space, 'y': labels, 'features': feature_names, 'labels': label_names}, file_handle)


def load_dataset_dump(filename):
    """Loads a compressed data set dump

    Parameters
    ----------

    filename : string
        Path to dump file, if without .bz2, the .bz2 extension will be appended.

    Returns
    -------

    data: dictionary {'X': array-like of array-likes, 'y': array-like of binary label vectors }
        The dictionary containing the data frame, with 'X' key storing the input space array-like of input feature vectors
        and 'y' storing labels assigned to each input vector, as a binary indicator vector (i.e. if 5th position has value 1
        then the input vector has label no. 5)

    """
    data = None

    if filename[-4:] != '.bz2':
        filename += ".bz2"

    with bz2.BZ2File(filename, "r") as file_handle:
        data = pickle.load(file_handle)

    return data
