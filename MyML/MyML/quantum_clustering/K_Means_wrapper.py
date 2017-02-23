import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

# Receives:
#  - mixture		:	n x d array with n points and d dimensions
#  - numClusters	:	number of clusters to use
#  - numInits		:	number of k-means runs
# Returns:
#  - k_centroids	:	list of final centroids of each iteration
#  - qk_assignment	:	list of assignments of each point to one of the 
#						centroids on each iteration
#  - k_timings_cg	:	list of timing for each iteration

def k_means(mixture, numClusters, numInits):
	'''This wrapper exists for comparison with QK-Means.'''

	k_timings_cg=list()
	start=datetime.now()

	k_assignment=list()
	k_centroids=list()
	k_inertia=list()

	for i in range(numInits):
		estimator = KMeans(n_clusters=numClusters, init='k-means++', n_init=1)
		assignment = estimator.fit_predict(mixture)
		centroids = estimator.cluster_centers_

		k_centroids.append(centroids)
		k_assignment.append(assignment)
		k_inertia.append(estimator.inertia_)

		k_timings_cg.append((datetime.now() - start).total_seconds())
		start=datetime.now()

	return k_centroids,k_assignment,k_timings_cg,k_inertia
