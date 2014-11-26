import arff
import bz2
import pickle
import numpy as np

class Dataset(object):
    @classmethod
    def load_arff_to_numpy(cls, filename, labelcount, endian = "big"):
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

        return input_space, labels.astype('i8')

    @classmethod
    def save_dataset_dump(cls, filename, input_space, labels):
        if filename[-4:] != '.bz2':
            filename += ".bz2"

        with bz2.BZ2File(filename, "wb") as file_handle:
            pickle.dump({'X': input_space, 'y': labels}, file_handle)

    @classmethod
    def load_dataset_dump(cls, filename):
        data = None

        if filename[-4:] != '.bz2':
            filename += ".bz2"

        with bz2.BZ2File(filename, "r") as file_handle:
            data = pickle.load(file_handle)
        
        return data