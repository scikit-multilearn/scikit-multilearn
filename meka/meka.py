import commands
import numpy as np
# import tempfile

class Meka(object):
    """docstring for MekaClassifier"""

    def __init__(self, meka_classifier = None, weka_classifier = None, java_command = '/usr/bin/java', meka_classpath = "/home/niedakh/icml/meka-1.7/lib/", threshold=0, verbosity=6):
        super(Meka, self).__init__()

        self.java_command = java_command
        self.classpath = meka_classpath
        self.meka_classifier = meka_classifier
        self.threshold = threshold
        self.verbosity = verbosity
        self.weka_classifier = weka_classifier
        self.output = None
        self.warnings = None
        self.results = None
        self.statistics = None

    def run(self, train_file, test_file):
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
        print(meka_command)
        status, output = commands.getstatusoutput(meka_command)
        
        if status != 0:
            raise Exception, output
        
        self.output = output
        self.parse_output()
        return self.results, self.statistics

    def parse_output(self):
        if self.output is None:
            self.results = None
            self.statistics = None
            return None

        predictions_split_head = "|==== PREDICTIONS ===============>\n"
        predictions_split_foot = "|==============================<\n"

        self.warnings = self.output.split(predictions_split_head)[0]

        # predictions, first split
        predictions = self.output.split(predictions_split_head)[1].split(predictions_split_foot)[0]
        # then clean up and remove empty lines
        predictions = filter(lambda x: len(x), predictions.replace('\n\n','\n').split('\n'))
        # parse into list of row classifications
        self.results = np.array(
            [map(lambda x: int(float(x)),
                 item.split('[ ')[2].split(' ]')[0].replace(',','.').split(' '))
             for item in predictions])

        # split, cleanup, remove empty lines
        statistics = self.output.split(predictions_split_head)[1].split(predictions_split_foot)[1]
        statistics = filter(lambda x: len(x), statistics.replace(' ','').replace('\n\n','\n').split('\n'))

        # remove per label stats, they can be calculated using python later, parse into a dict
        self.statistics = dict([item.split(':') for item in statistics if ']:' not in item])

