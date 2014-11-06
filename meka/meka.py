import commands
# import tempfile

class Meka(object):
    """docstring for MekaClassifier"""

    def __init__(self, meka_classifier=None, weka_classifier=None, threshold=0, verbosity=5,
                 meka_classpath="/home/niedakh/pwr/old/meka-1.5/lib/"):
        super(Meka, self).__init__()

        self.classpath = meka_classpath
        self.meka_classifier = meka_classifier
        self.threshold = threshold
        self.verbosity = verbosity
        self.weka_classifier = weka_classifier
        self.output = None
        self.warnings = None
        self.results = None
        self.statistics = None

    def run(self, train_file, test_file, input_instance_count, input_features_count, label_count):
        self.input_instance_count = input_instance_count
        self.input_features_count = input_features_count
        self.label_count = label_count
        self.output = None
        self.warnings = None

        # meka_command_string = 'java -cp "/home/niedakh/pwr/old/meka-1.5/lib/*" meka.classifiers.multilabel.MULAN -S RAkEL2  
        # -threshold 0 -t {train} -T {test} -verbosity {verbosity} -W weka.classifiers.bayes.NaiveBayes'
        # meka.classifiers.multilabel.LC, weka.classifiers.bayes.NaiveBayes
        meka_command_string = 'java -cp "{classpath}*" {meka} -threshold {threshold} -t {train} -T {test} -verbosity {verbosity} -W {weka}'

        input_files = {
            'meka': self.meka_classifier,
            'weka': self.weka_classifier,
            'train': train_file,
            'test': test_file,
            'threshold': self.threshold,
            'verbosity': self.verbosity,
            'classpath': self.classpath
        }
        meka_command = meka_command_string.format(**input_files)
        status, output = commands.getstatusoutput(meka_command)
        self.output = output
        self.parse_output()
        return self.results, self.statistics

    def parse_output(self):
        if self.output is None:
            self.results = None
            self.statistics = None
            return None

        split_position = self.output.find('\n    0') + 1
        self.output = self.output[split_position:]

        if split_position > 0:
            self.warnings = self.output[:split_position]

        self.parse_classification_results()
        self.parse_statistics()

    def parse_classification_results(self):
        self.results = [
            self.output.split('\n')[i].split('[')[2].strip('] ').split(' ')
            for i in range(self.input_instance_count)
        ]

        assert len(self.results) == self.input_instance_count


    def parse_statistics(self):
        self.statistics = dict(
            item.split(':') for item in
            self.output.replace(' ', '').replace('\n\n', '\n').split('\n')[self.input_instance_count:] if len(item) > 0
        )