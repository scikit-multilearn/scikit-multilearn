import arff
import bz2
import pickle
import numpy as np

from scipy import sparse


class Dataset(object):

    @classmethod
    def load_arff_to_numpy(cls, filename, labelcount, endian="big", input_feature_type='float', encode_nominal=True, load_sparse=False):
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

        data: dictionary {'X': scipy sparse matrix with input_feature_type elements, 'y': scipy sparse matrix of binary (int8) label vectors }
            The dictionary containing the data frame, with 'X' key storing the input space array-like of input feature vectors
            and 'y' storing labels assigned to each input vector, as a binary indicator vector (i.e. if 5th position has value 1
            then the input vector has label no. 5)

        """
        matrix = None
        if not load_sparse:
            arff_frame = arff.load(
                open(filename, 'rb'), encode_nominal=encode_nominal, return_type=arff.DENSE)
            matrix = sparse.csr_matrix(
                arff_frame['data'], dtype=input_feature_type)
        else:
            arff_frame = arff.load(
                open(filename, 'rb'), encode_nominal=encode_nominal, return_type=arff.COO)
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

        return X, y

    @classmethod
    def save_to_arff(cls, X, y, endian="little", save_sparse=True):
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

        save_sparse: boolean
            Whether to read arff file as a sparse file format, liac-arff breaks if sparse reading is enabled for non-sparse ARFFs.

        Returns
        -------

        data: dictionary {'X': scipy sparse matrix with input_feature_type elements, 'y': scipy sparse matrix of binary (int8) label vectors }
            The dictionary containing the data frame, with 'X' key storing the input space array-like of input feature vectors
            and 'y' storing labels assigned to each input vector, as a binary indicator vector (i.e. if 5th position has value 1
            then the input vector has label no. 5)

        """
        X = X.todok()
        y = y.todok()

        x_prefix = 0
        y_prefix = 0

        x_attributes = [(u'X{}'.format(i), u'NUMERIC')
                        for i in xrange(X.shape[1])]
        y_attributes = [(u'y{}'.format(i), [unicode(0), unicode(1)])
                        for i in xrange(y.shape[1])]

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
            data = [{} for r in xrange(X.shape[0])]
        else:
            data = [[0 for c in xrange(X.shape[1] + y.shape[1])]
                    for r in xrange(X.shape[0])]

        for keys, value in X.iteritems():
            data[keys[0]][x_prefix + keys[1]] = value

        for keys, value in y.iteritems():
            data[keys[0]][y_prefix + keys[1]] = value

        dataset = {
            u'description': u'traindata',
            u'relation': u'traindata: -C {}'.format(y.shape[1] * relation_sign),
            u'attributes': attributes,
            u'data': data
        }

        return arff.dumps(dataset)

    @classmethod
    def save_dataset_dump(cls, filename, input_space, labels):
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
        """
        if filename[-4:] != '.bz2':
            filename += ".bz2"

        with bz2.BZ2File(filename, "wb") as file_handle:
            pickle.dump({'X': input_space, 'y': labels}, file_handle)

    @classmethod
    def load_dataset_dump(cls, filename):
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
