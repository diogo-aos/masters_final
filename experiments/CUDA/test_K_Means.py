 # -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:06:18 2015

@author: Diogo Silva
"""

import numpy as np
from K_Means2 import *
from sklearn import datasets # generate gaussian mixture
from timeit import default_timer as timer # timing


##generate data
n = 1e6
d = 2
k = 20

n = np.int(n)

total_bytes = np.float((n * d + k * d + n * k) * 4)
print 'Memory used by arrays:\t',total_bytes/1024,'\tKBytes'
print '\t\t\t',total_bytes/(1024*1024),'\tMBytes'

print 'Memory used by data:  \t',n * d * 4 / 1024,'\t','KBytes'

## Generate data
#data = np.random.random((n,d)).astype(np.float32)
data, groundTruth = datasets.make_blobs(n_samples=n,n_features=d,centers=k,
                                        center_box=(-1000.0,1000.0))
data = data.astype(np.float32)

times=dict() #dicitonary to store times

start = timer()
grouperCUDA = K_Means()
grouperCUDA._cuda_mem = "manual"
grouperCUDA.fit(data,k,iters=3,mode="cuda")
times['cuda'] = timer() - start
print 'CUDA ','\t',times['cuda']

start = timer()
grouperNP = K_Means()
grouperNP.fit(data,k,iters=3,mode="numpy")
times['numpy'] = timer() - start
print 'NumPy','\t',times['numpy']

start = timer()
grouperP = K_Means()
grouperP.fit(data,k,iters=3,mode="python")
times['python'] = timer() - start



print 'Times'
print 'CUDA ','\t',times['cuda']
print 'NumPy','\t',times['numpy']
print 'Python','\t',times['python']


print "end"