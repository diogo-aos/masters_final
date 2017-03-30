import numpy as np
from MyML.cluster.K_Means3 import K_Means
from MyML.cluster.eac import EAC
from MyML.metrics import accuracy
from MyML.helper.partition import generateEnsemble
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from timeit import default_timer as timer # timing

cardinality = [1e3, 1e4, 1e5, 5e5, 1e6]
dimensionality = [2, 10, 100, 500, 1000]
n_parts = 30
centers = 6
rounds = 10

results = np.zeros((len(cardinality) * len(dimensionality) * rounds, 4), dtype = np.float64)
resultsIdx = 0

for n_samples in cardinality:
    for n_features in dimensionality:
        for r in range(rounds):
            print "n_samples: ", n_samples, "\tn_features: ", n_features
            n_samples = np.int(n_samples)

            #memRequired = n_samples * n_features * 4.0 / (1024 ** 2)
            generator = KMeans(init="random",n_init=1) # numpy generator
            # if memRequired > 500:
            #     generator = KMeans(init="random",n_init=1) # numpy generator
            # else:
            #     generator = K_Means() # cuda generator

            data, gt = make_blobs(n_samples = n_samples, n_features = n_features, centers = centers)
            data = data.astype(np.float32)

            start = timer()

            n_samples_sqrt = np.sqrt(n_samples)
            n_clusters = [n_samples_sqrt / 2, n_samples_sqrt]
            n_clusters = map(int,n_clusters)

            ensemble = generateEnsemble(data, generator, n_clusters, iters = 3)
            
            estimator = EAC(nsamples = n_samples,mat_sparse = False)
            estimator.fit(ensemble, files = False)

            elapsed = timer() - start

            results[resultsIdx,0] = n_samples
            results[resultsIdx,1] = n_features
            sparsity = (estimator._coassoc.nonzero()[0].size - np.float(n_samples)) / (n_samples**2) # sparsity
            results[resultsIdx,2] = sparsity
            results[resultsIdx,3] = elapsed
            resultsIdx += 1

            print "round: ", r, "\ttook: ", elapsed, " seconds", "\tsparsity:", sparsity
            
            np.savetxt("sparsityResults.csv", results, delimiter = ",")