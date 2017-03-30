# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:21:08 2015

@author: Diogo Silva

Description:

This test serves to test the NumPy performance on a vanilla installation.
It was written to be executed on a virtualenv with only Python + NumPy.
When running with Anaconda, NumPy gets optimized and gets access to Intel
MKL libraries.. For that reason, it becomes harder to understand the real
spedup of CUDA over NumPy.
"""

import numpy as np
from K_Means_noCUDA import *
from sklearn import datasets # generate gaussian mixture
from timeit import default_timer as timer # timing


##generate data
n = 2e6
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
grouperNP = K_Means()
grouperNP.fit(data,k,cuda=False)
times['numpy'] = timer() - start
#del grouperNP

print 'Times'
print 'NumPy','\t',times['numpy']