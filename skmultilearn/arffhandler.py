import arff
import numpy as np

class ArffHandler(object):

	@classmethod
	def load_arff_to_numpy(cls, filename, labelcount, endian = "big"):
	    arff_frame = arff.load(open(filename ,'rb'))
	    input_features_count = len(arff_frame['data'][0]) - labelcount
	    input_space = None
	    labels = None

	    if endian == "big":
	        input_space = np.array([row[labelcount:] for row in arff_frame['data']])
	        labels	    = np.array([row[:labelcount] for row in arff_frame['data']])
	    elif endian == "little":
	        input_space = np.array([row[:input_features_count] for row in arff_frame['data']])
	        labels      = np.array([row[-labelcount:] for row in arff_frame['data']])
	    else:
	        # unknown endian
	        return None

	    return input_space, labels.astype('i8')
