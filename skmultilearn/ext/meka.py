import subprocess
import numpy as np
import tempfile
import shlex
import scipy.sparse as sparse
import arff
import os


class Meka(object):
    """ Runs the MEKA classifier

        Parameters
        ----------

        meka_classifier : string
            The MEKA classifier string and parameters from the MEKA API, such as: "meka.classifiers.multilabel.MULAN -S RAkEL2"
        
        weka_classifier : string
            The WEKA classifier string and parameters from the WEKA API, such as: "weka.classifiers.trees.J48"
        
        java_command : string
            Path to test the java command

        meka_classpath: string
            Path to the MEKA class path folder, usually the folder lib in the directory MEKA was extracted to

    """
    def __init__(self, meka_classifier = None, weka_classifier = None, java_command = '/usr/bin/java', meka_classpath = "/home/niedakh/icml/meka-1.7/lib/"):
        super(Meka, self).__init__()

        self.java_command = java_command
        self.classpath = meka_classpath
        self.meka_classifier = meka_classifier
        self.verbosity = 5
        self.weka_classifier = weka_classifier
        self.output = None
        self.warnings = None
        self.clean()

    def save_to_arff(self, X, y, endian = "little", save_sparse = True):
    
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

        x_attributes = [(u'X{}'.format(i),u'NUMERIC') for i in xrange(X.shape[1])]
        y_attributes = [(u'y{}'.format(i), [unicode(0),unicode(1)]) for i in xrange(y.shape[1])]

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
            data = [[0 for c in xrange(X.shape[1] + y.shape[1])] for r in xrange(X.shape[0])]

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

    def clean(self):
        self.results = None
        self.statistics = None
        self.output = None
        self.error = None
        self.label_count = None
        self.instance_count = None


    def remove_temporary_files(self, temporary_files):
        for file_name in temporary_files:
            os.remove(file_name.name)

            arff_file_name = file_name.name + '.arff'
            if os.path.exists(arff_file_name):
                os.remove(arff_file_name)


    def run_meka_command(self, args):
        command_args = [
            self.java_command,
            '-cp', "{}*".format(self.classpath),
            self.meka_classifier,    
        ]

        if self.weka_classifier is not None:
            command_args += ['-W', self.weka_classifier]

        command_args += args

        meka_command = " ".join(command_args)

        pipes = subprocess.Popen(shlex.split(meka_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.output, self.error = pipes.communicate()

        if pipes.returncode != 0:
            raise Exception, self.output

    def fit(self, X, y):
        self.clean()
        self.label_count = y.shape[1]

        # we need this in case threshold needs to be recalibrated in meka
        self.train_data_ = self.save_to_arff(X, y)
        train_arff = tempfile.NamedTemporaryFile(delete = False)
        classifier_dump_file = tempfile.NamedTemporaryFile(delete = False)
        
        try:
            with open(train_arff.name + '.arff', 'w') as fp:
                fp.write(self.train_data_)

            input_args = [
                '-verbosity', "0",
                '-split-percentage', "100",
                '-t', train_arff.name + '.arff',
                '-d', classifier_dump_file.name,
            ]

            self.run_meka_command(input_args)

            self.classifier_dump = None

            with open(classifier_dump_file.name, 'rb') as fp:
                self.classifier_dump = fp.read()

        finally:
            self.remove_temporary_files([train_arff, classifier_dump_file])
            
        return self


    def predict(self, X):
        self.instance_count = X.shape[0]

        if self.classifier_dump is None:
            raise Exception, 'Not classified'

        sparse_y = sparse.coo_matrix((X.shape[0], self.label_count), dtype = int)

        try:
            train_arff = tempfile.NamedTemporaryFile(delete = False)
            test_arff = tempfile.NamedTemporaryFile(delete = False)
            classifier_dump_file = tempfile.NamedTemporaryFile(delete = False)

            with open(train_arff.name + '.arff', 'wb') as fp:
                fp.write(self.train_data_)

            with open(classifier_dump_file.name, 'wb') as fp:
                fp.write(self.classifier_dump)

            with open(test_arff.name + '.arff', 'wb') as fp:
                fp.write(self.save_to_arff(X, sparse_y))

            args = [
                '-l', classifier_dump_file.name
            ]

            self.run(train_arff.name + '.arff', test_arff.name + '.arff', args)
            self.parse_output()

        finally:
            self.remove_temporary_files([train_arff, test_arff, classifier_dump_file])
            
        return self.results

    def run(self, train_file, test_file, additional_arguments = []):
        """ Runs the meka classifiers

        Parameters
        ----------

        train_file : string
            Path to train .arff file in meka format (big endian, labels first in attributes list).
        
        test_file : string
            Path to test .arff file in meka format (big endian, labels first in attributes list).

        Returns
        -------

        predictions: sparse binary indicator matrix [n_test_samples, n_labels]
            array of binary label vectors including label predictions

        """
        self.output = None
        self.warnings = None

        # meka_command_string = 'java -cp "/home/niedakh/pwr/old/meka-1.5/lib/*" meka.classifiers.multilabel.MULAN -S RAkEL2  
        # -threshold 0 -t {train} -T {test} -verbosity {verbosity} -W weka.classifiers.bayes.NaiveBayes'
        # meka.classifiers.multilabel.LC, weka.classifiers.bayes.NaiveBayes

        args = [
            '-t', train_file,
            '-T', test_file,
            '-verbosity', str(5),
        ] + additional_arguments

        self.run_meka_command(args)
        return self

    def parse_output(self):
        """ Internal function for parsing MEKA output."""
        if self.output is None:
            self.results = None
            self.statistics = None
            return None

        predictions_split_head = '==== PREDICTIONS'
        predictions_split_foot = '|==========='

        if self.label_count is None:
            self.label_count = map(lambda y: int(y.split(')')[1].strip()), filter(lambda x: 'Number of labels' in x, self.output.split('\n')))[0]
        
        if self.instance_count is None:
            self.instance_count = int(float(filter(lambda x: '==== PREDICTIONS (N=' in x, self.output.split('\n'))[0].split('(')[1].split('=')[1].split(')')[0]))
        self.predictions = self.output.split(predictions_split_head)[1].split(predictions_split_foot)[0].split('\n')[1:-1]
        self.predictions = map(lambda z: map(lambda f: int(f.strip()),  z.split(',')), map(lambda y: y.split(']')[0], map(lambda x: x.split('] [')[1], self.predictions)))

        if self.verbosity == 6:
            self.results = sparse.csr_matrix(self.predictions)
        elif self.verbosity == 5:
            self.results = sparse.lil_matrix((self.instance_count, self.label_count), dtype='int')
            for row in xrange(self.instance_count):
                for label in self.predictions[row]:
                    self.results[row, label] = 1

        statistics = filter(lambda x: len(x)> 0 and '==' not in x, self.output.split('== Evaluation Info')[1].split('\n'))
        statistics = filter(lambda y: '  ' in y, map(lambda z: z.strip(), statistics))
        array_data = filter(lambda z: '[' in z, statistics)
        non_array_data = filter(lambda z: '[' not in z, statistics)
        
        self.statistics = {}
        for row in non_array_data:
            r = row.strip().split('  ')
            r = filter(lambda z: len(z) > 0, r)
            r = map(lambda z: z.strip(), r)
            if len(r) < 2:
                continue
            try:
                test_value = float(r[1])
            except ValueError:
                test_value = r[1]
            
            r[1] = test_value
            self.statistics[r[0]] = r[1]

        for row in array_data:
            r = row.strip().split('[')
            r = map(lambda z: z.strip(), r)
            r[1] = r[1].replace(', ',' ').replace(',','.').replace(']', '').split(' ')
            r[1] = filter(lambda x: len(x) > 0, r[1])
            self.statistics[r[0]] = r[1]

        return self.results, self.statistics


