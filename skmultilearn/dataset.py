import arff
import bz2
import pickle
import numpy as np

class Dataset(object):
    @classmethod
    def load_arff_to_numpy(cls, filename, labelcount, endian = "big", input_feature_type = 'float'):
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

        Returns
        -------
        
        data: dictionary {'X': array-like of array-likes, 'y': array-like of binary label vectors }
            The dictionary containing the data frame, with 'X' key storing the input space array-like of input feature vectors
            and 'y' storing labels assigned to each input vector, as a binary indicator vector (i.e. if 5th position has value 1
            then the input vector has label no. 5) 

        """
        arff_frame = arff.load(open(filename ,'rb'))
        input_features_count = len(arff_frame['data'][0]) - labelcount
        input_space = None
        labels = None

        if endian == "big":
            input_space = np.array([row[labelcount:] for row in arff_frame['data']])
            labels      = np.array([row[:labelcount] for row in arff_frame['data']])
        elif endian == "little":
            input_space = np.array([row[:input_features_count] for row in arff_frame['data']])
            labels      = np.array([row[-labelcount:] for row in arff_frame['data']])
        else:
            # unknown endian
            return None

        return input_space.astype(input_feature_type), labels.astype('i8')

    @classmethod
    def save_dataset_dump    (cls, filename, input_space, labels):
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