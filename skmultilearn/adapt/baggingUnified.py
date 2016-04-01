from skmultilearn.adapt.baggingNaive import BaggingNaive
import numpy, random
import itertools

class BaggingUnified(BaggingNaive):

    def __init__(self, classifier=None, require_dense=None, model_count=None):
        super(BaggingUnified, self).__init__(classifier=classifier, require_dense=require_dense)
        self.model_count=model_count


    def generate_partition(self, X, y):
        self.label_count = y.shape[1]
        self.instance_count = y.shape[0]
        y=y.toarray()
        table= [];
        labelIndex = 0
        while labelIndex <self.label_count :
            positiveLabels = [];
            instanceIndex = 0;
            while instanceIndex< self.instance_count:
                if y[instanceIndex][labelIndex] == 1:
                    positiveLabels.append(instanceIndex)
                instanceIndex+=1
            table.append(positiveLabels)
            labelIndex+=1

        instances_sets = []
        self.partition_size = int(numpy.ceil(self.instance_count / self.model_count))
        self.forOneLabel = int(numpy.ceil(self.partition_size / self.label_count))
        if(self.forOneLabel==0): self.forOneLabel = 5
        instances_sets =[]
        while (len(instances_sets) < self.model_count):
             i = 0
             set_for_labels = []
             while i< self.label_count:
                 if len(table[i])<self.forOneLabel:
                     set_for_labels.extend(table[i])
                 else:
                    set_for_labels.extend(random.sample(table[i], self.forOneLabel))
                 i+=1
             instances_sets.append(set_for_labels)
        #print (instances_sets)
        self.partition = instances_sets
