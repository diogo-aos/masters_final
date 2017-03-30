import numpy as numpy

def full2CSRGraph(assoc_mat):
	# use all the elements from both triangulars since we need edges
	# in both directions between each pair of vertices
	temp_mat = assoc_mat

	diag_inds = np.diag_indices_from(temp_mat)
	temp_mat[diag_inds] = 0 # zero the diagonal

	origin, dest = temp_mat.nonzero() # origin vertices are rows, destination are cols
	weights = temp_mat[origin, dest] # weights of edges are the values themselves

	firstedge = np.where(origin[1:]-origin[:-1] != 0)[0]
	firstedge = np.hstack(([0], firstedge, origin.size)) #indices of the first edge for each vertex
	# origin.size is included for the next step only, it's removed afterwards

	outdegree = firstedge[1:] - firstedge[:-1] # outdegree is the number of edges in each vertex

	firstedge = firstedge[:-1] #remove last element (number of vertices)

	return dest, weights, firstedge, outdegree
