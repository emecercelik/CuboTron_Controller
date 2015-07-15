import pickle
from numpy import linalg as LA
import numpy as np
from sklearn import linear_model
# =================================================================================================================================================
#                                       Functions

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def ActFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def NeuronFunc(x):
    return (1.+np.tanh(x-33.))*.5 # 4.=9 input

def InputFunc(x,dim,nAct,maxAcc,minAcc):
    rangeAcc=maxAcc-minAcc
    off=np.arange(minAcc+rangeAcc/(2.*nAct),maxAcc,rangeAcc/nAct)
    res=np.array([[ActFunc(nAct,maxAcc,minAcc,x[j],off[i]) for i in range(len(off))] for j in range(len(x))])
    res=res.reshape(res.size,1)
    for i in range(res.size):
        if res[i]<1e-2:
            res[i]=.0
    return res

############################################################################################

a=GetPickle('paramWalking1')
b=a[2:,:-4]
c=a[2:,-4:]

# First regression type
wout=c.T.dot(b)
inverse=LA.inv(b.T.dot(b)+.00001*np.random.rand(b.shape[1],b.shape[1]))+1e-8*np.identity(int(b.shape[1]))
wout=wout.dot(inverse)

# Second Regression Type
clf=linear_model.Ridge(alpha=.5)
clf.fit(b,c)

#Save
PickleIt(clf.coef_,'wout')
PickleIt(clf.intercept_,'intercept')
