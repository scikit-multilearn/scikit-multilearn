#copyright @Fernando Benites

from ..base import MLClassifierBase

import numpy.core.umath as umath
import scipy.sparse
import numpy

class Neuron:
    def __init__(self,startpoint,label):
        #vector must be in complement form
        self.vc =  startpoint 
#        ones = scipy.ones(startpoint.shape);
#        self.vc=numpy.concatenate((startpoint, ones - startpoint))
        self.label=label

class MLARAM(MLClassifierBase):
    """Multi-label ARAM   classifier. See http://dx.doi.org/10.1109/ICDMW.2015.14"""
    BRIEFNAME = "ML-ARAM"
    def __init__(self,classifier = None, vigilance=0.9,threshold=0.02, printouttst=0,tneurons=None, tdebug=0, require_dense = True):
        super(MLARAM, self).__init__(classifier, require_dense)
        
        if tneurons!=None:
            self.neurons=tneurons
        else:
            self.neurons=[]
        self.labels=[]
        self.vigilance=vigilance
        self.threshold=threshold
	self.printouttst=printouttst
        self.allneu=""
        self.debug=tdebug
        self.online=1
        self.copyable_attrs = ["classifier", "require_dense", "vigilance","threshold","online"]
        
    def reset(self):
        self.labels=[]
        self.neurons=[]
    #@profile
    def fit(self,X,y):
        
        labdict = {}
        if len(self.neurons) == 0:
            ones = scipy.ones(X[0].shape);
            self.neurons.append(Neuron(numpy.concatenate((X[0], ones - X[0]), 1),y[0]))
            startc = 1
            labdict[y[0].nonzero()[0].tostring()] = [0]
        else:
            startc = 0
        newlabel = 0
        import time
        time1=time.time()
        ones = scipy.ones(X[0].shape);
        for i1,f1 in enumerate(X[startc: ], startc):
            #print i1,X.shape[0],len(self.neurons), newlabel,len(labdict) 
            if i1%1000==0:
                print i1,X.shape[0],len(self.neurons), newlabel, "time ",time.time()-time1
                time1=time.time()
            found=0
            if scipy.sparse.issparse(f1):
                f1=f1.todense()
            fc = numpy.concatenate((f1, ones - f1), 1)
                
            activationn = [0] * len(self.neurons)
            activationi = [0] * len(self.neurons)
            ytring=y[i1].nonzero()[0].tostring()
            if ytring in labdict:
                fcs = fc.sum()
                for i2 in labdict[ytring]:
                    minnfs = umath.minimum(self.neurons[i2].vc, fc).sum()
                    activationi[i2] =minnfs/fcs
                    activationn[i2] =minnfs/self.neurons[i2].vc.sum()
#            for i2, n1 in enumerate(self.neurons):                
#                 if (y[i1] - n1.label).sum() == 0:
#                     activation[i2] =umath.minimum(n1.vc, fc).sum()/fc.sum()


                     #activation[i2] = numpy.concatenate((umath.minimum(n1.down,f1),umath.minimum(ones-n1.up,ones-f1)),1).sum()/float(numpy.concatenate((f1,ones-f1),1).sum())
            

            if numpy.max(activationn) == 0:
                newlabel += 1
                self.neurons.append(Neuron(fc,y[i1]))
                labdict.setdefault(ytring, []). append(len(self.neurons) - 1)
           

                continue
            inds = numpy.argsort(activationn)
            
            # if activationi[inds[ - 1]] < self.vigilance:
            indc = numpy.where(numpy.array(activationi)[inds[::-1]]>self.vigilance)[0]
            if indc.shape[0] == 0: 
                self.neurons.append(Neuron(fc,y[i1]))
                
                labdict.setdefault(ytring, []). append(len(self.neurons) - 1)
                continue
                

            winner =inds[::- 1][indc[0]]
            self.neurons[winner].vc= umath.minimum(self.neurons[winner].vc,fc)
            #print i1,winner, len(self.neurons),self.neurons[winner].vc.sum()

            #if winner==407:
            #    print i1
            labadd = numpy.zeros(y[0].shape,dtype=y[0].dtype)
            labadd[y[i1].nonzero()] = 1
            self.neurons[winner].label +=   labadd
            
        
    
    #@profile
    def predict(self,X):
        ft = 0
        tt=0
        fn = 0
        tp=0
        fp=0
        mrec=0
        mpre=0
        macf1=0
        result = []
        if len(X) == 0: 
            return
        ones = scipy.ones(X[0].shape);
        n1s = [0] *  len(self.neurons)
        allranks = []
        neuronsactivated=[]
        #for i in range(0, len(self.neurons)):
        #    n1s[i] = self.neurons[i].vc.sum()

        #if self.debug==1:
        #    import time
        #    start = time.time()
        #allneu=numpy.concatenate(([n1.vc for n1 in self.neurons]))
        allneu=numpy.vstack([n1.vc for n1 in self.neurons])
        allneusum=allneu.sum(1)
        #if self.debug==1:            
        #    elapsed = (time.time() - start)
        #    print "for calculation of big neurons matrix ",elapsed
        import time
        time1=time.time()
        for i1,f1 in enumerate(X):
            if self.printouttst==1:
                print i1,
            if (i1%10)+1==10:
                #print i1
                print i1,time.time()-time1
                time1=time.time()

            if scipy.sparse.issparse(f1):

                f1 = f1.todense()
            fc = numpy.concatenate((f1, ones - f1), 1)
            # numpy with umath broadcast itself: http://www.scipy.org/EricsBroadcastingDoc
            #fca=numpy.tile(fc,(len(self.neurons),1))
            #activity=[0] * len(self.neurons)
            #for i2,n1 in enumerate(self.neurons):
            #    activity[i2] =umath.minimum(n1.vc, fc).sum()/n1s[i2]
                # this does NOT make a copy of a, simply indexes it backward, and should
            #if self.debug==1:
            #    start = time.time()
            activity=(umath.minimum(fc,allneu).sum(1)/allneusum).squeeze().tolist()[0]
            
            # be very fast
            sortedact=numpy.argsort(activity)[::-1]
            
            #if self.debug==1:            
            #    elapsed = (time.time() - start)
            #    print "for activation and sorting  ",elapsed

            winner=sortedact[0]
            diff_act=activity[winner]-activity[sortedact[-1]]
            #print i1,winner
            

            largest_activ = 1;

            par_t=self.threshold
            for i in range(1, len(self.neurons)):
                activ_change = (activity[winner]-activity[sortedact[i]])/activity[winner];
                if activ_change >par_t*diff_act:
                    break

                largest_activ +=  1;

            # print self.neurons[winner].label,labels, winner

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
            allranks.append(rank)
            neuronsactivated.append((actives,activity_actives))
                
        return numpy.array(numpy.matrix(result))

    #@profile
    def predict_proba(self,X):
        ft = 0
        tt=0
        fn = 0
        tp=0
        fp=0
        mrec=0
        mpre=0
        macf1=0
        result = []
        if len(X) == 0: 
            return
        ones = scipy.ones(X[0].shape);
        n1s = [0] *  len(self.neurons)
        allranks = []
        neuronsactivated=[]
        #for i in range(0, len(self.neurons)):
        #    n1s[i] = self.neurons[i].vc.sum()

        #if self.debug==1:
        #    import time
        #    start = time.time()
        #allneu=numpy.concatenate(([n1.vc for n1 in self.neurons]))
        allneu=numpy.vstack([n1.vc for n1 in self.neurons])
        allneusum=allneu.sum(1)
        #if self.debug==1:            
        #    elapsed = (time.time() - start)
        #    print "for calculation of big neurons matrix ",elapsed
        import time
        time1=time.time()
        for i1,f1 in enumerate(X):
            if self.printouttst==1:
                print i1,
            if (i1%10)+1==10:
                #print i1
                print i1,time.time()-time1
                time1=time.time()

            if scipy.sparse.issparse(f1):

                f1 = f1.todense()
            fc = numpy.concatenate((f1, ones - f1), 1)
            # numpy with umath broadcast itself: http://www.scipy.org/EricsBroadcastingDoc
            #fca=numpy.tile(fc,(len(self.neurons),1))
            #activity=[0] * len(self.neurons)
            #for i2,n1 in enumerate(self.neurons):
            #    activity[i2] =umath.minimum(n1.vc, fc).sum()/n1s[i2]
                # this does NOT make a copy of a, simply indexes it backward, and should
            #if self.debug==1:
            #    start = time.time()
            activity=(umath.minimum(fc,allneu).sum(1)/allneusum).squeeze().tolist()[0]
            
            # be very fast
            sortedact=numpy.argsort(activity)[::-1]
            
            #if self.debug==1:            
            #    elapsed = (time.time() - start)
            #    print "for activation and sorting  ",elapsed

            winner=sortedact[0]
            diff_act=activity[winner]-activity[sortedact[-1]]
            #print i1,winner
            

            largest_activ = 1;

            par_t=self.threshold
            for i in range(1, len(self.neurons)):
                activ_change = (activity[winner]-activity[sortedact[i]])/activity[winner];
                if activ_change >par_t*diff_act:
                    break

                largest_activ +=  1;

            # print self.neurons[winner].label,labels, winner

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