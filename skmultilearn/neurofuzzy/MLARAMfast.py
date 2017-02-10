from builtins import range
from builtins import object
#copyright @Fernando Benites

from ..base import MLClassifierBase

import numpy.core.umath as umath
import scipy.sparse
import numpy

class Neuron(object):
    def __init__(self,startpoint,label):
        #vector must be in complement form
        self.vc =  startpoint 
#        ones = scipy.ones(startpoint.shape);
#        self.vc=numpy.concatenate((startpoint, ones - startpoint))
        self.label=label

class MLARAM(MLClassifierBase):
    """Multi-label ARAM   classifier. See http://dx.doi.org/10.1109/ICDMW.2015.14

    Parameters
    ----------

    vigilance : vigilance parameter for adaptiv resonance theory networks, controls how large a hyperbox can be, 1 it is small (no compression), 0 should assume all range. Normally set between 0.8 and 0.999, it is dataset dependent. It is responsible for the creation of the prototypes, therefore training of the network.
    threshold : controls how many prototypes participate by the prediction, can be changed at the testing phase.
    tneurons  : if the network should inherited neurons (prototypes) from another network
    tdebug : set debug modus

    Whether the base classifier requires input as dense arrays, False by default"""
    BRIEFNAME = "ML-ARAM"

    def __init__(self, vigilance=0.9,threshold=0.02, tneurons=None):
        super(MLARAM, self).__init__()
        
        if tneurons!=None:
            self.neurons=tneurons
        else:
            self.neurons=[]
        self.labels=[]
        self.vigilance=vigilance
        self.threshold=threshold
	
        self.allneu=""
        self.online=1
        self.alpha=0.0000000000001
        self.copyable_attrs += ["neurons", "labels", "vigilance","threshold", "allneu", "online", "alpha"]
        
    def reset(self):
        self.labels=[]
        self.neurons=[]

    #@profile
    def fit(self,X,y):
        
        labdict = {}
        if len(X[0].shape)==1:
            ismatrix=0
        else:
            ismatrix=1
        xma=X.max()
        xmi=X.min()
        if xma<0 or xma>1 or xmi<0 or xmi>1:
            X=numpy.multiply(X-xmi,1/(xma-xmi))
            
        if len(self.neurons) == 0:
            ones = scipy.ones(X[0].shape)
            self.neurons.append(Neuron(numpy.concatenate((X[0], ones - X[0]), ismatrix),y[0]))
            startc = 1
            labdict[y[0].nonzero()[0].tostring()] = [0]
        else:
            startc = 0
        newlabel = 0
        ones = scipy.ones(X[0].shape)
        for i1,f1 in enumerate(X[startc: ], startc):
            found=0
            if scipy.sparse.issparse(f1):
                f1=f1.todense()
            fc = numpy.concatenate((f1, ones - f1), ismatrix)
                
            activationn = [0] * len(self.neurons)
            activationi = [0] * len(self.neurons)
            ytring=y[i1].nonzero()[0].tostring()
            if ytring in labdict:
                fcs = fc.sum()
                for i2 in labdict[ytring]:
                    minnfs = umath.minimum(self.neurons[i2].vc, fc).sum()
                    activationi[i2] =minnfs/fcs
                    activationn[i2] =minnfs/self.neurons[i2].vc.sum()
            

            if numpy.max(activationn) == 0:
                newlabel += 1
                self.neurons.append(Neuron(fc,y[i1]))
                labdict.setdefault(ytring, []). append(len(self.neurons) - 1)
           

                continue
            inds = numpy.argsort(activationn)
            
            indc = numpy.where(numpy.array(activationi)[inds[::-1]]>self.vigilance)[0]
            if indc.shape[0] == 0: 
                self.neurons.append(Neuron(fc,y[i1]))
                
                labdict.setdefault(ytring, []). append(len(self.neurons) - 1)
                continue
                

            winner =inds[::- 1][indc[0]]
            self.neurons[winner].vc= umath.minimum(self.neurons[winner].vc,fc)
            

            
            labadd = numpy.zeros(y[0].shape,dtype=y[0].dtype)
            labadd[y[i1].nonzero()] = 1
            self.neurons[winner].label +=   labadd
            
        
    
    #@profile
    def predict(self,X):
        result=[]
        ranks=self.predict_proba(X)
        for rank in ranks:
            sortedRankarg = numpy.argsort(-rank)
            diffs=-numpy.diff([rank[k] for k in sortedRankarg])
            
            indcutt=numpy.where(diffs==(diffs).max())[0]
            if len(indcutt.shape)==1:
                indcut=indcutt[0]+1
            else:
                indcut=indcutt[0,-1]+1
            label=numpy.zeros(rank.shape)

            label[sortedRankarg[0:indcut]]=1
                
            result.append(label)
                
        return numpy.array(numpy.matrix(result))

    #@profile
    def predict_proba(self,X):
        result = []
        if len(X) == 0: 
            return
        if len(X[0].shape)==1:
            ismatrix=0
        else:
            ismatrix=1
        xma=X.max()
        xmi=X.min()
        if xma<0 or xma>1 or xmi<0 or xmi>1:
            X=numpy.multiply(X-xmi,1/(xma-xmi))
        ones = scipy.ones(X[0].shape);
        n1s = [0] *  len(self.neurons)
        allranks = []
        neuronsactivated=[]

        allneu=numpy.vstack([n1.vc for n1 in self.neurons])
        allneusum=allneu.sum(1)+self.alpha


        for i1,f1 in enumerate(X):
            if scipy.sparse.issparse(f1):

                f1 = f1.todense()
            fc = numpy.concatenate((f1, ones - f1), ismatrix)
            activity=(umath.minimum(fc,allneu).sum(1)/allneusum).squeeze().tolist()
            if ismatrix==1:
                activity=activity[0]
            
            # be very fast
            sortedact=numpy.argsort(activity)[::-1]
            

            winner=sortedact[0]
            diff_act=activity[winner]-activity[sortedact[-1]]

            

            largest_activ = 1

            par_t=self.threshold
            for i in range(1, len(self.neurons)):
                activ_change = (activity[winner]-activity[sortedact[i]])/activity[winner];
                if activ_change >par_t*diff_act:
                    break

                largest_activ +=  1

            rbsum = sum([activity[k] for k in sortedact[0:largest_activ]])

            rank = activity[winner]*self.neurons[winner].label
            actives =[]
            activity_actives =[]
            actives.append(winner)
            activity_actives.append(activity[winner])
            for i in range(1,largest_activ):
                rank+=activity[sortedact[i]]*self.neurons[sortedact[i]].label
                actives.append(sortedact[i])
                activity_actives.append(activity[sortedact[i]])
            rank/= rbsum
            allranks.append(rank)
                
        return numpy.array(numpy.matrix(allranks))
