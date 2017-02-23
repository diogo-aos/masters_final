import numpy as np
from MyML.cluster.K_Means3 import K_Means, _cu_label_kernel_dists
from MyML.cluster.eac import EAC
from MyML.metrics import accuracy
from MyML.helper.partition import generateEnsemble
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from numbapro import cuda

import sys

n_samples = 1e4
n_features = 2
centers = 6
n_samples = np.int(n_samples)
data, gt = make_blobs(n_samples = n_samples, n_features = n_features, centers = centers)
data = data.astype(np.float32)
n_samples_sqrt = np.int(np.sqrt(n_samples))
n_clusters = [n_samples_sqrt / 2, n_samples_sqrt]
n_clusters = map(int,n_clusters)

generator = K_Means(cuda_mem="manual")

generator.n_clusters = 176
generator.max_iter = 1

generator.N = n_samples
generator.D = n_features

generator.centroids = generator._init_centroids(data)
generator.labels = cuda.pinned_array(shape=generator.N, dtype = np.int32)
generator._dists = cuda.pinned_array(shape=generator.N, dtype = np.float32)

generator._compute_cuda_dims(data)
gridDim = generator._gridDim
blockDim = generator._blockDim

print "grid: ", gridDim
print "block: ", blockDim

dData = cuda.to_device(data)

dCentroids = cuda.to_device(generator.centroids)

dLabels = cuda.device_array_like(generator.labels)
dDists = cuda.device_array_like(generator._dists)

startE = cuda.event()
endE = cuda.event()

startE.record()
_cu_label_kernel_dists[gridDim,blockDim](dData,dCentroids,dLabels,dDists)
endE.record()
endE.synchronize()
print cuda.event_elapsed_time(startE,endE)

startE.record()
dDists.copy_to_host(ary = generator._dists)
labels = dLabels.copy_to_host(ary = generator.labels)
endE.record()
endE.synchronize()
print cuda.event_elapsed_time(startE,endE)

del generator, dData, dCentroids, dLabels, dDists

generator = K_Means(mode="numpy")
generator.n_clusters = 176
generator.max_iter = 1
generator.fit(data)

generator = K_Means(mode="cuda", cuda_mem="manual")
generator.n_clusters = 176
generator.max_iter = 1
generator.fit(data)

if __name__ = "__main__":
	main()