import numpy as np
from numba import jit, njit

from scipy_numba.spatial.distance import squareform

from sparse import EAC_CSR

class EAC_FULL():

    def __init__(self, n_samples, dtype=np.uint8, condensed=False, **kwargs):
        self.n_samples = n_samples
        self.dtype = dtype

        self.nnz = 0
        self.n_updates = 0

        self.condensed = condensed

        self._validate_func_options()

        self.is_diassoc = False

        if self.condensed:
            n = sum(xrange(1, n_samples))
            self.coassoc = np.zeros(n, dtype=dtype)
            self.update_partition = self._update_partition_condensed
        else:
            self.coassoc = np.zeros((n_samples, n_samples), dtype=dtype)
            self.update_partition = self._update_partition_full

        # the matrix will usually be created with the knowledge of the
        # number of partitions
        self.max_val = kwargs.get("max_val", None) 

    def _validate_func_options(self):
        if type(self.n_samples) != int:
            raise TypeError("number of samples must be an integer")
        if self.n_samples <= 0:
            raise ValueError("number of samples must be an integer above 0")

        if not issubclass(self.dtype, (np.integer, np.float)):
            raise TypeError("dtype must be a NumPy integer or float type")

        if self.condensed not in (True,False):
            raise ValueError("condensed must be boolean")

    def update_ensemble(self, ensemble):
        for p in xrange(len(ensemble)):
            self.update_partition(ensemble[p])

    def _update_partition_full(self, partition):
        for cluster in partition:
            numba_update_coassoc_with_cluster(self.coassoc, cluster)
        self.n_updates += 1

    def _update_partition_condensed(self, partition):
        for cluster in partition:
            numba_update_condensed_coassoc_with_cluster(self.coassoc,
                                                        cluster,
                                                        self.n_samples)
        self.n_updates += 1        

    def todense(self):
        if self.condensed:
            return squareform(self.coassoc)
        else:
            return self.coassoc.copy()

    def tocsr(self):
        if self.condensed:
            return csr_matrix(squareform(self.coassoc))
        else:
            return csr_matrix(self.coassoc)

    def get_degree(self):
        n = self.n_samples

        degree = np.zeros(self.n_samples, dtype=np.int32)
        if not self.condensed:
            nnz = full_get_assoc_degree(self.coassoc, degree)
        else:
            nnz = full_condensed_assoc_degree(self.coassoc, degree, n)
        self.degree = degree
        self.nnz = nnz

    def make_diassoc(self, max_val=None):
        if max_val is None:
            if self.max_val is not None:
                max_val = self.max_val + 1
            else:
                max_val = self.coassoc.max() + 1

        if self.condensed:
            make_diassoc_1d(self.coassoc, max_val)
        else:
            make_diassoc_2d(self.coassoc, max_val)

        self.is_diassoc = True

    def __eq__(self, other):

        if isinstance(other, EAC_FULL):
            me_condensed = self.condensed
            other_condensed = other.condensed

            me_cmp_mat = self.coassoc
            if not me_condensed:
                me_cmp_mat = squareform(self.coassoc)
            other_cmp_mat = other.coassoc
            if not other_condensed:
                other_cmp_mat = squareform(other.coassoc)

            return (me_cmp_mat == other_cmp_mat).all()

        elif isinstance(other, EAC_CSR):
            me_condensed = self.condensed
            other_condensed = other.condensed

            me_cmp_mat = self.tocsr()
            other_cmp_mat = other.tocsr()
            if other_condensed:
                other_cmp_mat = other_cmp_mat + other_cmp_mat.T

            return (me_cmp_mat == other_cmp_mat).all() 

        else:
            print other.__class__
            raise TypeError("Comparison only with EAC_CSR or EAC_FULL objects.")
    
    def __ne__(self, other):
        not self.__eq__(other)


# 2d
@njit
def make_diassoc_2d(ary, val):
    for row in range(ary.shape[0]):
        for col in range(ary.shape[1]):
            tmp = ary[row,col]
            ary[row,col] = val - tmp

#1d
@njit
def make_diassoc_1d(ary, val):
    for i in range(ary.size):
        tmp = ary[i]
        ary[i] = val - tmp


# - - - - - - - - - -  UPDATE COASSOCS FUNCTIONS  - - - - - - - - - - 

@njit
def numba_update_coassoc_with_cluster(coassoc, cluster):
    """
    Receives the coassoc 2-d array and the cluster 1-d array. 
    """
    for i in range(cluster.size-1):
        curr_i = cluster[i]
        for j in range(i+1, cluster.size):
            curr_j = cluster[j]
            if i == j:
                continue
            coassoc[curr_i, curr_j] += 1
            coassoc[curr_j, curr_i] += 1

@njit
def numba_update_condensed_coassoc_with_cluster(coassoc, cluster, n):
    """
    Receives the condensed coassoc 1-d array and the cluster 1-d array. 
    """
    for i in range(cluster.size-1):
        curr_i = cluster[i]
        for j in range(i+1, cluster.size):
            curr_j = cluster[j]
            idx = condensed_index(n, curr_i, curr_j)
            coassoc[idx] += 1

@njit
def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.

    Source: SciPy project
    """
    if i < j:
        return n * i - (i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) / 2) + (i - j - 1)
    else:
        return -1

# - - - - - - - - - -  METADATA FUNCTIONS  - - - - - - - - - - 

@njit
def full_get_assoc_degree(ary, degree):
    """
    Function will fill the degree array with the number of nonzero values in
    each row, such that degree[i] contains the number of nonzero values of
    the i-th row of the ary matrix.
    Inputs:
        ary     : input matrix of shape r,c
        degree  : array of shape r
    Outputs:
        nnz     : total number of nonzero values
    """
    rows, cols = ary.shape
    nnz = 0
    for row in range(rows):
        row_deg = 0
        for col in range(cols):
            if ary[row,col] != 0:
                row_deg += 1
        degree[row] = row_deg
        nnz += row_deg
    return nnz

@njit
def full_condensed_assoc_degree(ary, degree, n):
    idx = 0
    row = 0
    nnz = 0
    for i in range(n-1,0,-1): # i is the length of each row
        row_deg = 0
        for j in range(i): # j is iterating over the cols
            if ary[idx] != 0:
                row_deg += 1
            idx += 1
        degree[row] = row_deg
        nnz += row_deg
        row += 1
    return nnz

def full_condensed_assoc_degree_np(ary, degree, n):
    idx=0
    row = 0
    for i in xrange(n-1, 0, -1):
        degree[row] = ary[idx:idx+i].nonzero()[0].size
        idx+=i
        row += 1



@njit
def numba_array2d_nnz(ary, width, height):
    """
    Function will return the number of nonzero values of the full matrix.
    Inputs:
        ary     : input matrix
        width   : number of columns of the matrix
        height  : number of rows of the matrix
    Outputs:
        nnz     : number of nonzero values
    """
    nnz = 0
    for line in range(height):
        for col in range(width):
            if ary[line,col] != 0:
                nnz = nnz + 1
    return nnz

def get_max_assocs_in_sample(assoc_mat):
    """
    Returns the maximum number of co-associations a sample has and the index of
    that sample.
    """
    max_row_size=0
    max_row_idx=-1
    row_idx=0
    for row in assoc_mat:
        if row.nonzero()[0].size > max_row_size:
            max_row_size = row.nonzero()[0].size
            max_row_idx = row_idx
        row_idx += 1
        
    return max_row_size, max_row_idx

# - - - - - - - - - -  POST-PROCESSING FUNCTIONS  - - - - - - - - - - 

def apply_threshold_to_coassoc(threshold, max_val, assoc_mat):
    """
    threshold   : all co-associations whose value is below 
                  threshold * max_val are zeroed
    max_val     : usually number of partitions
    assoc_mat   : co-association matrix
    """
    assoc_mat[assoc_mat < threshold * max_val] = 0


def coassoc_to_condensed_diassoc(assoc_mat, max_val, copy=False):
    """
    Simple routine to tranform a full square co-association matrix in a 
    condensed form diassociation matrix. Max val is the value to use for
    normalization - usually the number of partitions. The diassociation
    matrix will have no zeros - minimum value possible is 1.
    """

    if copy:
        assoc_mat_use = assoc_mat.copy()
    else:
        assoc_mat_use = assoc_mat
    
    make_diassoc_2d(assoc_mat_use, max_val) # make matrix diassoc
    fill_diag(assoc_mat_use, 0) # clear diagonal

    condensed_diassoc = squareform(assoc_mat_use)

    return condensed_diassoc

if __name__ == '__main__':
    print('add tests')
