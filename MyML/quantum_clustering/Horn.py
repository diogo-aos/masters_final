import numpy as np
from sklearn import preprocessing
from datetime import datetime

def _corr(C):
	R = np.empty_like(C)
	#compute correlation from covariance
	for i,ci in enumerate(C):
		for j,cij in enumerate(ci):
			R[i,j] = cij / np.sqrt(C[i,i] * C[j,j])
	return R

def pcaFun(x, whiten=False, e=0, type='cov', method='svd',
			center=True, normalize=False):
	# x 		:	n x m numpy.array of n points and m dimensions
	# whiten	:	boolean parameter - whiten data or not
	# e 		:	normalization parameter for whitening data

	n,d = x.shape
	oX=x
	# normalize
	if normalize:
		x=sklearn.normalize(x,axis=0)

	# center data
	if center:
		avg = np.mean(x,axis=0)
		x = x - avg

	if method == 'eig':
		# compute covariance matrix
		if type == 'cov':
			C = x.T.dot(x)
			C /= n
		elif type == 'corr':
			#C=np.corrcoef(x,rowvar=0, bias=1)
			C = x.T.dot(x)
			C /= n
			C = _corr(C)
		else:
			raise Exception('Incompatible argument value \
				\'type='+str(type)+'\'')

		# compute eig
		eigVals,eigVect = np.linalg.eig(C)

		#sort eigenthings
		eigValOrder = eigVals.argsort()[::-1] #descending eigen indeces
		
		sortedEigVect = np.zeros(eigVect.shape)
		sortedEigVal = np.zeros(eigVals.shape)

		for i,j in enumerate(eigValOrder):
			sortedEigVect[:,i] = eigVect[:,j]
			sortedEigVal[i] = eigVals[j]

		comps = sortedEigVect
		eigs = sortedEigVal

	elif method == 'svd':
		U,S,V = np.linalg.svd(x)
		comps = V.T
		eigs = (S**2) / n
	else:
		raise Exception('Incompatible argument value \
				\'method='+str(method)+'\'')

	# project data
	projX = x.dot(comps)

	if whiten is True:
		whiten_vect = np.sqrt((eigs + e))
		projX = projX / whiten_vect

	return projX, comps, eigs


# function graddesc(xyData,q,[steps])
# purpose: performing quantum clustering in and moving the 
#          data points down the potential gradient
# input: xyData - the data vectors
#        q=a parameter for the parsen window variance (q=1/(2*sigma^2))
#		 sigma=parameter for the parsen window variance (choose q or sigma)
#        steps=number of gradient descent steps (default=50)
#		 eta=gradient descent step size
# output: D=location of data o=point after GD 

def graddesc(xyData, **kwargs):
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	# 				Argument treatment
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	steps = kwargs.get('steps', 50)
	sigma = kwargs.get('sigma', 0.1)
	q = kwargs.get('q', (1.0 / (2 * pow(sigma, 2))))
	D = kwargs.get('r', xyData)
	eta = kwargs.get('eta'. 0.1)
	return_eta = kwargs.get('return_eta', False)
	timeit = kwargs.get('timeit', False)
	timelapse = kwargs.get('timelapse', False)
	all_square = kwargs.get('all_square', False)

	if all_square is not False:
		if xyData.shape[1]>2:
			raise Exception('all_square should not be used in data > 2 dims')
		points = kwargs['all_square']
		totalPoints = pow(kwargs['all_square'],2)
		a = np.linspace(-1,1,points)
		D = [(x,y) for x in a for y in a]
		D = np.array(D)

	if timelapse:
		tD = list()
		timelapse_count = 0
		if 'timelapse_list' in argKeys:
			timelapse_list = kwargs['timelapse_list']
		elif 'timelapse_percent' in argKeys:
			timelapse_percent = kwargs['timelapse_percent']
			list_inc = int(steps/(steps*timelapse_percent))
			if list_inc == 0:
				list_inc = 1
			timelapse_list = range(steps)[::list_inc]
		else:
			timelapse_percent = 0.25
			list_inc = int(steps/(steps*timelapse_percent))
			if list_inc == 0:
				list_inc = 1
			timelapse_list = range(steps)[::list_inc]
			timelapse_list = range(steps)[::int(steps*timelapse_percent)]

	if timeit:
		#timings=np.zeros(steps+1) #+1 for the total time
		timings = datetime.now()


	# add more states to timelapse list
	if timelapse:
		if timelapse_count in timelapse_list:
			tD.append(D)
		timelapse_count += 1


	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	# 				Algorithm starts here
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

		
	# first run
	V,P,E,dV = qc(xyData, q=q, r=D)

	for j in range(4):
		for i in range(steps/4):
			# normalize potential gradient
			dV = preprocessing.normalize(dV)
			
			# gradient descent
			D = D - eta*dV

			# add more states to timelapse list
			if timelapse:
				if timelapse_count in timelapse_list:
					tD.append(D)
				timelapse_count += 1		

			"""	
			if timeit:
				start_time=datetime.now()"""

			# perform Quantum Clustering
			V,P,E,dV = qc(xyData,q=q,r=D)

			"""
			if timeit:
				timeings[i*4]=(datetime.now() - start).total_seconds()"""
		eta *= 0.5



	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	# 				Algorithm ends here
	# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	if timeit:
		timings = (datetime.now() - timings).total_seconds()

	if timelapse:
		tD.append(D)
		D = tD

	returnList = [D,V,E]

	if return_eta:
		returnList.append(eta)

	if timeit:
		returnList.append(timings)

	#returnList.append(timelapse_list)
	return returnList
	

# function qc (matlab doc)
# purpose: performing quantum clustering in n dimensions
# input:
#       ri - a vector of points in n dimensions
#       q - the factor q which determines the clustering width
#       r - the vector of points to calculate the potential for. equals ri if not specified
# output:
#       V - the potential
#       P - the wave function
#       E - the energy
#       dV - the gradient of V
# example: [V,P,E,dV] = qc ([1,1;1,3;3,3],5,[0.5,1,1.5]);
# see also: qc2d

def qc(ri,**kwargs):
	argKeys=kwargs.keys()

	if 'q' in argKeys:
		q = kwargs['q']
	elif 'sigma' in argKeys:
		sigma = kwargs['sigma']
		q = 1 / (2 * pow(sigma,2))
	else:
		sigma = 0.1
		q = 1 / (2 * pow(sigma,2))

	if 'r' in argKeys:
		r = kwargs['r']
	else:
		r = ri

	pointsNum,dims = ri.shape
	calculatedNum = r.shape[0]

	# prepare the potential
	V = np.zeros(calculatedNum)
	dP2 = np.zeros(calculatedNum)

	# prepare P
	P = np.zeros(calculatedNum)
	singledV1 = np.zeros((calculatedNum,dims))
	singledV2 = np.zeros((calculatedNum,dims))

	dV1 = np.zeros((calculatedNum,dims))
	dV2 = np.zeros((calculatedNum,dims))
	dV = np.zeros((calculatedNum,dims))

	# prevent division by zero
	# calculate V
	# run over all the points and calculate for each the P and dP2

	for point in range(calculatedNum):

		# compute ||x-xi||^2
		# axis=1 will sum rows instead of columns
		D2 = np.sum(pow(r[point]-ri,2),axis=1)

		# compute gaussian
		singlePoint = np.exp(-q*D2)

		# compute Laplacian of gaussian = ||x-xi||^2 * exp(...)
		singleLaplace = D2 * singlePoint

		#compute gradient components
		aux = r[point] - ri
		for d in range(dims):
			singledV1[:,d] = aux[:,d] * singleLaplace
			singledV2[:,d] = aux[:,d] * singlePoint

		P[point] = np.sum(singlePoint)
		dP2[point] = np.sum(singleLaplace)
		dV1[point] = np.sum(singledV1,axis=0)
		dV2[point] = np.sum(singledV2,axis=0)

	# if there are points with 0 probability, 
	# assigned them the lowest probability of any point
	P = np.where(P==0, np.min(np.extract((P!=0), P)), P)

	# compute ground state energy
	V = -dims/2 + q*dP2 / P
	E = -min(V)

	# compute potential on points
	V += E

	# compute gradient of V
	for d in range(dims):
		dV[:,d] = -q * dV1[:,d] + (V-E+(dims+2)/2) * dV2[:,d]

	return V, P, E, dV


# clust=fineCluster(xyData,minD) cluster xyData points when closer than minD
# output: clust=vector the cluter index that is asigned to each data point
#        (it's cluster serial #)
def fineCluster(xyData, minD, potential=None, timeit=False):
	
	if potential is not None:
		usePotential = True
	else:
		usePotential = False

	n = xyData.shape[0]
	clust = np.zeros(n)

	if timeit:
		timings = datetime.now()

	if usePotential:
		# index of points sorted by potential
		sortedUnclust = potential.argsort()

		# index of unclestered point with lowest potential
		i = sortedUnclust[0]
	else:
		i = 0

	# fist cluster index is 1
	clustInd = 1

	while np.min(clust) == 0:
		x = xyData[i]

		# euclidean distance from ith point to others
		D = np.sum(pow(xyData-x,2),axis=1)
		D = pow(D,0.5)

		clust = np.where(D<minD,clustInd,clust)
		
		# index of non clustered points
		# unclust=[x for x in clust if x == 0]
		clusted = clust.nonzero()[0]

		if usePotential:
			# sorted index of non clustered points
			sortedUnclust = [x for x in sortedUnclust if x not in clusted]

			if len(sortedUnclust) == 0:
				break

			#index of unclustered point with lowest potential
			i = sortedUnclust[0]
		else:
			#index of first unclustered datapoint
			i = np.argmin(clust)

		clustInd += 1

	if timeit:
		timings = (datetime.now()-timings).total_seconds()

	returnList = [clust]
	if timeit:
		return clust, timings

	return clust