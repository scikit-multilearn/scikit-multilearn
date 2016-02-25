import subprocess
import numpy as np
# import tempfile
import shlex
import scipy.sparse as sparse

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
        self.threshold = 0
        self.verbosity = 6
        self.weka_classifier = weka_classifier
        self.output = None
        self.warnings = None
        self.results = None
        self.statistics = None

    def run(self, train_file, test_file):
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
        meka_command_string = '{java} -cp "{classpath}*" {meka} -threshold {threshold} -t {train} -T {test} -verbosity {verbosity} -W {weka}'
        

        input_files = {
            'java': self.java_command,
            'meka': self.meka_classifier,
            'weka': self.weka_classifier,
            'train': train_file,
            'test': test_file,
            'threshold': self.threshold,
            'verbosity': self.verbosity,
            'classpath': self.classpath
        }
        meka_command = meka_command_string.format(**input_files)
        pipes = subprocess.Popen(shlex.split(meka_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = pipes.communicate()

        if pipes.returncode != 0:
            raise Exception, output
        
        self.output = output
        self.parse_output()
        return self.results

    def parse_output(self):
        """ Internal function for parsing MEKA output."""
        if self.output is None:
            self.results = None
            self.statistics = None
            return None

        predictions_split_head = '==== PREDICTIONS'
        predictions_split_foot = '|==========='

        self.label_count = map(lambda y: int(y.split(')')[1].strip()), filter(lambda x: 'Number of labels' in x, self.output.split('\n')))
        self.instance_count = int(float(filter(lambda x: '==== PREDICTIONS (N=' in x, self.output.split('\n'))[0].split('(')[1].split('=')[1].split(')')[0]))
        self.predictions = self.output.split(predictions_split_head)[1].split(predictions_split_foot)[0].split('\n')[1:-1]


        if self.verbosity == 6:
            self.results = sparse.csr_matrix(map(lambda x: map(lambda y: int(float(y)), x.split('] [ ')[1].replace(',','.').replace(' ]','').split(' ')), self.predictions))
        elif self.verbosity == 5:
            dane = map(lambda x: x.strip().split(' '), predictions)
            self.results = sparse.lil_matrix((self.instance_count, self.label_count))
            for row in dane:
                i = int(row[0])-1
                for j in eval(row[2]):
                    self.results[i,j] =1 