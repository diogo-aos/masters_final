import unittest

import numpy as np
from scipy.sparse.csr import csr_matrix
from numba import cuda, njit, jit, int32, float32

from MyML.utils.profiling import Timer


path_4elt = 'datasets/4elt.npz'


def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


@njit
def outdegree_from_firstedge(firstedge, outdegree, n_edges):
    n_vertices = firstedge.size
    for v in range(n_vertices - 1):
        outdegree[v] = firstedge[v + 1] - firstedge[v]
    outdegree[n_vertices - 1] = n_edges - firstedge[n_vertices - 1]

class TestStringMethods(unittest.TestCase):

  def test_boruvka_sequential(self):

    sp_mat = load_sparse_csr(path_4elt)
    dest = sp_mat.indices
    weight = sp_mat.data
    firstedge = sp_mat.indptr[:-1]  # last 
    outdegree = np.emptylike(firstedge)
    outdegree_from_firstedge(firstedge, outdegree, dest.size)

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
    t1.tac()

    print "mst size", mst.size

    if n_edges < mst.size:
        mst = mst[:n_edges]

    print "time elapsed: ", t1.elapsed
    mst.sort() # mst has to be sorted for comparison with device mst because different threads might be able to write first
    print mst
    print n_edges    
    

if __name__ == '__main__':
    unittest.main()



