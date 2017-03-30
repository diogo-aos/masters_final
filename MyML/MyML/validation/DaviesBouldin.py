import numpy as np

class DaviesBouldin:
	'''Initialize with the data, centroids, assignment and degree of norm to use (by default L2).
	Use _eval()_ method to compute score.
	'''

	def __init__(self, data, centroids, assignment, p=2):
		# expecting data as numpy.array of D dimensions (columns) and N data points (rows) 
		## AND centroid index in last column
		# expecting centroids as numpy.array of D dimensions and K centroids
		self.data = data
		self.centroids = centroids
		self.assignment = assignment

		self.numClusters = centroids.shape[0]
		self.numData = data.shape[0]
		self.normOrder = p

	def intra(self):
		S = np.zeros(self.numClusters) #cluster scores for each cluster
		counter = np.zeros(self.numClusters) #counter of points in each cluster

		#computing the intra-cluster scores
		for i in range(0,self.numData):
			clusterIndex = self.assignment[i]

			# data point minus corresponding centroid
			dist = self.data[i,:]-self.centroids[clusterIndex,:]

			# compute l2 norm
			S[clusterIndex] += np.linalg.norm(dist,self.normOrder)

			# increment datapoint counter for each cluster
			counter[clusterIndex] += 1

		# compute final cluster score
		self.S = S / counter

	def inter(self):
		M = np.zeros((self.numClusters,self.numClusters))

		# the M matrix will be symmetrical
		for i in range(0,self.numClusters-1):
			for j in range(i+1,self.numClusters):
				M[i,j] = np.linalg.norm(self.centroids[i,:]-self.centroids[j,:],self.normOrder)
				M[j,i] = M[i,j]

		self.M = M / self.numClusters

	def totalScore(self):
		R=np.zeros((self.numClusters,self.numClusters))
		
		# the M matrix will be symmetrical
		for i in range(0,self.numClusters-1):
			for j in range(i+1,self.numClusters):
				R[i,j] = (self.S[i] + self.S[j]) / self.M[i,j]
				R[j,i] = R[i,j]

		self.R=R


	def computeDB(self):
		# select highest R for each cluster (1st dimension chooses cluster, 2nd dim has R)
		D = np.amax(self.R,1)

		self.DB = np.sum(D)



	def eval(self):

		self.intra()
		self.inter()
		self.totalScore()
		self.computeDB()

		return self.DB