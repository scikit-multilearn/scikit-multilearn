import arff
import bz2
import pickle
from scipy import sparse
import hashlib

import os
import requests
import shutil
from collections import defaultdict


def get_data_home(data_home=None, subdirectory=''):
    """Return the path of the scikit-multilearn data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the :code:`data_home` is set to a folder named
    :code:`'scikit_ml_learn_data'` in the user home folder.

    Alternatively, it can be set by the :code:`'SCIKIT_ML_LEARN_DATA'`
    environment variable or programmatically by giving an explicit
    folder path. The :code:`'~'` symbol is expanded to the user home
    folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str (default is None)
        the path to the directory in which scikit-multilearn data sets
        should be stored, if None the path is generated as stated above

    subdirectory : str, default ''
        return path subdirectory under data_home if data_home passed or under default if not passed

    Returns
    --------
    str
        the path to the data home
    """
    if data_home is None:
        if len(subdirectory) > 0:
            data_home = os.environ.get('SCIKIT_ML_LEARN_DATA', os.path.join('~', 'scikit_ml_learn_data', subdirectory))
        else:
            data_home = os.environ.get('SCIKIT_ML_LEARN_DATA', os.path.join('~', 'scikit_ml_learn_data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str (default is None)
        the path to the directory in which scikit-multilearn data sets
        should be stored.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def _get_download_base_url():
    """Returns base URL for data sets."""

    return 'http://scikit.ml/datasets/'


def available_data_sets():
    """Lists available data sets and their variants

    Returns
    -------
    dict[(set_name, variant_name)] -> [md5, file_name]
        available datasets and their variants with the key pertaining
        to the :code:`(set_name, variant_name)` and values include md5 and file name on server
    """
    r = requests.get(_get_download_base_url() + 'data.list')
    if r.status_code != 200:
        r.raise_for_status()
    else:
        raw_data_list = r.text

        variant_information = defaultdict(list)
        for row in raw_data_list.split('\n'):
            md5, file_name = row.split(';')
            set_name, variant = file_name.split('.')[0].split('-')
            if (set_name, variant) in variant_information:
                raise Exception('Data file broken, files doubled, please file bug report.')
            variant_information[(set_name, variant)] = [md5, file_name]
        return variant_information


def download_dataset(set_name, variant, data_home=None):
    """Downloads a data set

    Parameters
    ----------
    set_name : str
        name of set from :func:`available_data_sets`
    variant : str
        variant of the data set from :func:`available_data_sets`
    data_home : default None, str
        custom base folder for data, if None, default is used

    Returns
    -------
    str
        path to the downloaded data set file on disk
    """

    data_sets = available_data_sets()
    if (set_name, variant) not in data_sets:
        raise ValueError('The set {} in variant {} does not exist on server.'.format(set_name, variant))

    md5, name = data_sets[set_name, variant]

    if data_home is None:
        target_name = os.path.join(get_data_home(), name)
    else:
        target_name = os.path.join(data_home, name)

    if os.path.exists(target_name):
        if md5 == _get_md5(target_name):
            print ("{}:{} - exists, not redownloading".format(set_name, variant))
            return target_name
        else:
            print ("{}:{} - exists, but MD5 sum mismatch - redownloading".format(set_name, variant))
    else:
        print("{}:{} - does not exists downloading".format(set_name, variant))

    # not found or broken md5
    _download_single_file(name, target_name)
    found_md5 = _get_md5(target_name)
    if md5 != found_md5:
        raise Exception(
            "{}: MD5 mismatch {} vs {} - possible download error".format(name, md5, found_md5))

    print("Downloaded {}-{}".format(set_name, variant))

    return target_name


def load_dataset(set_name, variant, data_home=None):
    """Loads a selected variant of the given data set

    Parameters
    ----------
    set_name : str
        name of set from :func:`available_data_sets`
    variant : str
        variant of the data set
    data_home : default None, str
        custom base folder for data, if None, default is used

    Returns
    --------
    dict
        the loaded multilabel data set variant in the scikit-multilearn
        format, see data_sets
    """

    path = download_dataset(set_name, variant, data_home)
    if path is not None:
        return load_dataset_dump(path)

    return None


def load_from_arff(filename, label_count, label_location="end",
                   input_feature_type='float', encode_nominal=True, load_sparse=False,
                   return_attribute_definitions=False):
    """Method for loading ARFF files as numpy array

    Parameters
    ----------
    filename : str
        path to ARFF file
    labelcount: integer
        number of labels in the ARFF file
    endian: str {"big", "little"} (default is "big")
        whether the ARFF file contains labels at the beginning of the
        attributes list ("start", MEKA format)
        or at the end ("end", MULAN format)
    input_feature_type: numpy.type as string (default is "float")
        the desire type of the contents of the return 'X' array-likes,
        default 'i8', should be a numpy type,
        see http://docs.scipy.org/doc/numpy/user/basics.types.html
    encode_nominal: bool (default is True)
        whether convert categorical data into numeric factors - required
        for some scikit classifiers that can't handle non-numeric
        input features.
    load_sparse: boolean (default is False)
        whether to read arff file as a sparse file format, liac-arff
        breaks if sparse reading is enabled for non-sparse ARFFs.
    return_attribute_definitions: boolean (default is False)
        whether to return the definitions for each attribute in the
        dataset

    Returns
    -------
    X : :mod:`scipy.sparse.lil_matrix` of `input_feature_type`, shape=(n_samples, n_features)
        input feature matrix
    y : :mod:`scipy.sparse.lil_matrix` of `{0, 1}`, shape=(n_samples, n_labels)
        binary indicator matrix with label assignments
    names of attributes : List[str]
        list of attribute names from ARFF file
    """

    if not load_sparse:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.DENSE
        )
        matrix = sparse.csr_matrix(
            arff_frame['data'], dtype=input_feature_type
        )
    else:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.COO
        )
        data = arff_frame['data'][0]
        row = arff_frame['data'][1]
        col = arff_frame['data'][2]
        matrix = sparse.coo_matrix(
            (data, (row, col)), shape=(max(row) + 1, max(col) + 1)
        )

    if label_location == "start":
        X, y = matrix.tocsc()[:, label_count:].tolil(), matrix.tocsc()[:, :label_count].astype(int).tolil()
        feature_names = arff_frame['attributes'][label_count:]
        label_names = arff_frame['attributes'][:label_count]
    elif label_location == "end":
        X, y = matrix.tocsc()[:, :-label_count].tolil(), matrix.tocsc()[:, -label_count:].astype(int).tolil()
        feature_names = arff_frame['attributes'][:-label_count]
        label_names = arff_frame['attributes'][-label_count:]
    else:
        # unknown endian
        return None

    if return_attribute_definitions:
        return X, y, feature_names, label_names
    else:
        return X, y


def save_to_arff(X, y, label_location="end", save_sparse=True, filename=None):
    """Method for dumping data to ARFF files

    Parameters
    ----------
    X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
        input feature matrix
    y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
        binary indicator matrix with label assignments
    label_location: string {"start", "end"} (default is "end")
        whether the ARFF file will contain labels at the beginning of the
        attributes list ("start", MEKA format)
        or at the end ("end", MULAN format)
    save_sparse: boolean
        Whether to save in ARFF's sparse dictionary-like format instead of listing all
        zeroes within file, very useful in multi-label classification.
    filename : str or None
        Path to ARFF file, if None, the ARFF representation is returned as string
    Returns
    -------
    str or None
        the ARFF dump string, if filename is None
    """
    X = X.todok()
    y = y.todok()

    x_prefix = 0
    y_prefix = 0

    x_attributes = [(u'X{}'.format(i), u'NUMERIC')
                    for i in range(X.shape[1])]
    y_attributes = [(u'y{}'.format(i), [str(0), str(1)])
                    for i in range(y.shape[1])]

    if label_location == "end":
        y_prefix = X.shape[1]
        relation_sign = -1
        attributes = x_attributes + y_attributes

    elif label_location == "start":
        x_prefix = y.shape[1]
        relation_sign = 1
        attributes = y_attributes + x_attributes

    else:
        raise ValueError("Label location not in {start, end}")

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

    arff_data = arff.dumps(dataset)

    if filename is None:
        return arff_data

    with open(filename, 'w') as fp:
        fp.write(arff_data)


def save_dataset_dump(input_space, labels, feature_names, label_names, filename=None):
    """Saves a compressed data set dump

    Parameters
    ----------
    input_space: array-like of array-likes
        Input space array-like of input feature vectors
    labels: array-like of binary label vectors
        Array-like of labels assigned to each input vector, as a binary
        indicator vector (i.e. if 5th position has value 1
        then the input vector has label no. 5)
    feature_names: array-like,optional
        names of features
    label_names: array-like, optional
        names of labels
    filename : str, optional
        Path to dump file, if without .bz2, the .bz2 extension will be
        appended.
    """
    data = {'X': input_space, 'y': labels, 'features': feature_names, 'labels': label_names}

    if filename is not None:
        if filename[-4:] != '.bz2':
            filename += ".bz2"

        with bz2.BZ2File(filename, "wb") as file_handle:
            pickle.dump(data, file_handle)
    else:
        return data


def load_dataset_dump(filename):
    """Loads a compressed data set dump

    Parameters
    ----------
    filename : str
        path to dump file, if without .bz2 ending, the .bz2 extension will be appended.

    Returns
    -------
    X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
        input feature matrix
    y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
        binary indicator matrix with label assignments
    names of attributes: List[str]
        list of attribute names for `X` columns
    names of labels: List[str]
        list of label names for `y` columns
    """

    if not os.path.exists(filename):
        raise IOError("File {} does not exist, use load_dataset to download file".format(filename))

    if filename[-4:] != '.bz2':
        filename += ".bz2"

    with bz2.BZ2File(filename, "r") as file_handle:
        data = pickle.load(file_handle)

    return data['X'], data['y'], data['features'], data['labels']


def _download_single_file(data_file_name, target_file_name, base_url=None):
    base_url = base_url or _get_download_base_url()
    r = requests.get(base_url + data_file_name, stream=True)
    if r.status_code == 200:
        with open(target_file_name, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        r.raise_for_status()


def _get_md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
