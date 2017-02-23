
# coding: utf-8

# author: Diogo Silva

# In[1]:


# In[23]:

import numpy as np
from MyML.cluster.K_Means3 import K_Means, _cu_label_kernel_dists
from MyML.cluster.eac import EAC
from MyML.metrics import accuracy
from MyML.helper.partition import generateEnsemble
from sklearn.datasets.samples_generator import make_blobs 
from numbapro import cuda

# In[3]:

cardinality = [1e3, 1e4, 1e5, 5e5, 1e6]
dimensionality = [2, 10, 100, 500, 1000]
n_parts = 30
centers = 6



n_samples = 1e5
n_features = 100
n_samples = np.int(n_samples)
data, gt = make_blobs(n_samples = n_samples, n_features = n_features, centers = centers)
data = data.astype(np.float32)
n_samples_sqrt = np.int(np.sqrt(n_samples))
n_clusters = [n_samples_sqrt / 2, n_samples_sqrt]
n_clusters = map(int,n_clusters)

# generator = K_Means(cuda_mem="manual")
# generator.n_clusters = n_clusters[-1]
# generator.max_iter = 3
# generator.fit(data)

generator = K_Means(cuda_mem="manual")	
generator.N = n_samples
generator.D = n_features
generator.n_clusters = n_clusters[-1]

generator.centroids = generator._init_centroids(data)
generator.labels = np.empty(shape=generator.N, dtype = np.int32)
generator._dists = np.empty(shape=generator.N, dtype = np.float32)

generator._compute_cuda_dims(data)
gridDim = generator._gridDim
blockDim = generator._blockDim

dData = cuda.to_device(data)
dCentroids = cuda.to_device(generator.centroids)

dLabels = cuda.device_array_like(generator.labels)
dDists = cuda.device_array_like(generator._dists)

_cu_label_kernel_dists[gridDim,blockDim](dData,dCentroids,dLabels,dDists)

dDists.copy_to_host(ary = generator._dists)
labels = dLabels.copy_to_host(ary = generator.labels)