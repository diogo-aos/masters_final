# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:32:02 2015

@author: Diogo Silva
"""

import numpy as np
from K_Means import *
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

grouper = K_Means(N=n,D=d,K=k)
centroids = grouper._init_centroids(data)

times = dict()


# Distance matrix
start = timer()
dist_mat_np = grouper._np_calc_dists(data,centroids)
times["dist_mat np"] = timer() - start

start = timer()
dist_mat_cu_auto = grouper._cu_calc_dists(data,centroids,gridDim=None,
                                     blockDim=None,memManage='auto',
                                     keepDataRef=False)
times["dist_mat cuda manual"] = timer() - start

start = timer()
dist_mat_cu_man  = grouper._cu_calc_dists(data,centroids,gridDim=None,
                                     blockDim=None,memManage='auto',
                                     keepDataRef=False)
times["dist_mat cuda auto"] = timer() - start


print "Distance matrix"
print "Numpy == CUDA Man:    ",'\t', np.allclose(dist_mat_np,
                                                 dist_mat_cu_man)
print "Numpy == CUDA Auto:   ",'\t', np.allclose(dist_mat_np,
                                                 dist_mat_cu_auto)
print "CUDA Auto == CUDA Man:",'\t', np.allclose(dist_mat_cu_auto,
                                                 dist_mat_cu_man)



# Assignment and grouped data
start = timer()
assign,groupedData = grouper._assign_data(data,dist_mat_cu_man)
times["assign and group"] = timer() - start



# Centroid calculation
start = timer()
computedCentroids = grouper._np_recompute_centroids(groupedData)
times["centroid computation"] = timer() - start

print computedCentroids

print "Times"
for k,v in times.iteritems():
    print k,': ',v