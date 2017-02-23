# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD, (C) 2011

import numpy as np
# cimport numpy as np

import os.path
import tables

import numba as nb
from MyML.utils.sorting import csr_datasort, data_argmax

from scipy.sparse import csr_matrix, isspmatrix_csc, isspmatrix
from scipy.sparse.csgraph._validation import validate_graph
# from scipy_numba.sparse.csgraph._validation import validate_graph

#include 'parameters.pxi'

ITYPE = np.int32

def minimum_spanning_tree(csgraph, overwrite=False):
    r"""
    minimum_spanning_tree(csgraph, overwrite=False)
    Return a minimum spanning tree of an undirected graph
    A minimum spanning tree is a graph consisting of the subset of edges
    which together connect all connected nodes, while minimizing the total
    sum of weights on the edges.  This is computed using the Kruskal algorithm.
    .. versionadded:: 0.11.0
    Parameters
    ----------
    csgraph : array_like or sparse matrix, 2 dimensions
        The N x N matrix representing an undirected graph over N nodes
        (see notes below).
    overwrite : bool, optional
        if true, then parts of the input graph will be overwritten for
        efficiency.
    Returns
    -------
    span_tree : csr matrix
        The N x N compressed-sparse representation of the undirected minimum
        spanning tree over the input (see notes below).
    Notes
    -----
    This routine uses undirected graphs as input and output.  That is, if
    graph[i, j] and graph[j, i] are both zero, then nodes i and j do not
    have an edge connecting them.  If either is nonzero, then the two are
    connected by the minimum nonzero value of the two.
    Examples
    --------
    The following example shows the computation of a minimum spanning tree
    over a simple four-component graph::
         input graph             minimum spanning tree
             (0)                         (0)
            /   \                       /
           3     8                     3
          /       \                   /
        (3)---5---(1)               (3)---5---(1)
          \       /                           /
           6     2                           2
            \   /                           /
             (2)                         (2)
    It is easy to see from inspection that the minimum spanning tree involves
    removing the edges with weights 8 and 6.  In compressed sparse
    representation, the solution looks like this:
    >>> from scipy.sparse import csr_matrix
    >>> from scipy.sparse.csgraph import minimum_spanning_tree
    >>> X = csr_matrix([[0, 8, 0, 3],
    ...                 [0, 0, 2, 5],
    ...                 [0, 0, 0, 6],
    ...                 [0, 0, 0, 0]])
    >>> Tcsr = minimum_spanning_tree(X)
    >>> Tcsr.toarray().astype(int)
    array([[0, 0, 0, 3],
           [0, 0, 2, 5],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    global NULL_IDX
    
    # csgraph = validate_graph(csgraph, True, csgraph.dtype, dense_output=False,
    #                          copy_if_sparse=not overwrite)
    N = csgraph.shape[0]

    data = csgraph.data
    indices = csgraph.indices
    indptr = csgraph.indptr

    rank = np.zeros(N, dtype=ITYPE)
    predecessors = np.arange(N, dtype=ITYPE) # this array is used to keep track of which tree each vertex belongs to

    i_sort = np.argsort(data)
    row_indices = np.zeros(len(data), dtype=ITYPE)

    iters = _min_spanning_tree(data, indices, indptr, i_sort,
                       row_indices, predecessors, rank)

    sp_tree = csr_matrix((data, indices, indptr), (N, N))
    sp_tree.eliminate_zeros()

    return sp_tree

@nb.njit
def _min_spanning_tree(data,
                       col_indices,
                       indptr,
                       i_sort,
                       row_indices,
                       predecessors,
                       rank):
    # Work-horse routine for computing minimum spanning tree using
    #  Kruskal's algorithm.  By separating this code here, we get more
    #  efficient indexing.
    # cdef unsigned int i, j, V1, V2, R1, R2, n_edges_in_mst, n_verts
    # cdef DTYPE_t E
    n_verts = predecessors.shape[0]
    
    # Arrange `row_indices` to contain the row index of each value in `data`.
    # Note that the array `col_indices` already contains the column index.
    for i in range(n_verts):
        for j in range(indptr[i], indptr[i + 1]):
            row_indices[j] = i
    
    # step through the edges from smallest to largest.
    #  V1 and V2 are the vertices, and E is the edge weight connecting them.
    n_edges_in_mst = 0
    i = 0
    while i < i_sort.shape[0] and n_edges_in_mst < n_verts - 1:
        j = i_sort[i]
        V1 = row_indices[j]
        V2 = col_indices[j]
        E = data[j]

        # progress upward to the head node of each subtree
        R1 = V1
        while predecessors[R1] != R1:
            R1 = predecessors[R1]
        R2 = V2
        while predecessors[R2] != R2:
            R2 = predecessors[R2]

        # Compress both paths.
        while predecessors[V1] != R1:
            predecessors[V1] = R1
        while predecessors[V2] != R2:
            predecessors[V2] = R2
            
        # if the subtrees are different, then we connect them and keep the
        # edge.  Otherwise, we remove the edge: it duplicates one already
        # in the spanning tree.
        if R1 != R2:
            n_edges_in_mst += 1
            
            # Use approximate (because of path-compression) rank to try
            # to keep balanced trees.
            if rank[R1] > rank[R2]:
                predecessors[R2] = R1
            elif rank[R1] < rank[R2]:
                predecessors[R1] = R2
            else:
                predecessors[R2] = R1
                rank[R1] += 1
        else:
            data[j] = 0
        
        i += 1

    iters = i

    # We may have stopped early if we found a full-sized MST so zero out the rest
    while i < i_sort.shape[0]:
        j = i_sort[i]
        data[j] = 0
        i += 1

    return iters

def minimum_spanning_tree_lean(csgraph, overwrite=False):
    r"""Lean memory variation of the normal algorithm. The difference lies in 
    the fact that no argsort is performed.
    """
    global NULL_IDX
    
    # csgraph = validate_graph(csgraph, True, csgraph.dtype, dense_output=False,
    #                          copy_if_sparse=not overwrite)
    N = csgraph.shape[0]

    data = csgraph.data
    indices = csgraph.indices
    indptr = csgraph.indptr

    rank = np.zeros(N, dtype=ITYPE)
    predecessors = np.arange(N, dtype=ITYPE) # this array is used to keep track of which tree each vertex belongs to

    csr_datasort(data, indices, indptr, N)
    degree = indptr[1:] - indptr[:-1]

    mst_idx = np.full(N-1, dtype=np.int64)
    mst_rows = np.full(N-1, dtype=np.int32)

    _min_spanning_tree_lean(data, indices, indptr, degree,
                       predecessors, rank, mst_idx, mst_rows)

    sp_tree = csr_matrix((data, indices, indptr), (N, N))
    sp_tree.eliminate_zeros()

    return sp_tree

@nb.njit
def _min_spanning_tree_lean(data,
                       col_indices,
                       indptr,
                       degree,
                       predecessors,
                       rank,
                       mst_idx, mst_rows,
                       row_max, ):
    # Work-horse routine for computing minimum spanning tree using
    #  Kruskal's algorithm.  By separating this code here, we get more
    #  efficient indexing.
    # cdef unsigned int i, j, V1, V2, R1, R2, n_edges_in_mst, n_verts
    # cdef DTYPE_t E
    n_verts = predecessors.size
    n_edges = data.size
    
    # step through the edges from smallest to largest.
    #  V1 and V2 are the vertices, and E is the edge weight connecting them.
    n_edges_in_mst = 0
    i = 0
    while i < n_edges and n_edges_in_mst < n_verts - 1:
        j, row = data_argmax(data, indptr, degree)

        V1 = row
        V2 = col_indices[j]
        E = data[j]

        # progress upward to the head node of each subtree
        R1 = V1
        while predecessors[R1] != R1:
            R1 = predecessors[R1]
        R2 = V2
        while predecessors[R2] != R2:
            R2 = predecessors[R2]

        # Compress both paths.
        while predecessors[V1] != R1:
            predecessors[V1] = R1
        while predecessors[V2] != R2:
            predecessors[V2] = R2
            
        # if the subtrees are different, then we connect them and keep the
        # edge.  Otherwise, we remove the edge: it duplicates one already
        # in the spanning tree.
        if R1 != R2:
            mst_idx[n_edges_in_mst] = j
            mst_rows[n_edges_in_mst] = row

            n_edges_in_mst += 1
            
            # Use approximate (because of path-compression) rank to try
            # to keep balanced trees.
            if rank[R1] > rank[R2]:
                predecessors[R2] = R1
            elif rank[R1] < rank[R2]:
                predecessors[R1] = R2
            else:
                predecessors[R2] = R1
                rank[R1] += 1
        else:
            data[j] = 0

        # decrement the degree of row so the retrieved association is not 
        # considered again; note that the j assoc. belongs to row
        degree[row] -= 1
        i += 1

@nb.njit
def data_argmax(data, indptr, degree):
    arg = -1
    val = -1
    row = -1
    i=0
    n = indptr.size-1
    for i in xrange(i, n):
        i_degree = degree[i]
        if i_degree == 0:
            continue
        i_indptr = indptr[i]
        carg = i_indptr + i_degree - 1
        cval = data[carg]
        if cval > val:
            arg = carg
            val = cval
            row = i
    return arg, row


# TODO
# add folder exist check
# add windows + linux folder support

def minimum_spanning_tree_disk(csgraph, overwrite=False,
                               arrays_dir='./', index_dir='/tmp/',
                               table_file=None):
    global NULL_IDX
    
    # csgraph = validate_graph(csgraph, True, csgraph.dtype, dense_output=False,
    #                          copy_if_sparse=not overwrite)
    N = csgraph.shape[0] # number of vertices

    data = csgraph.data
    indices = csgraph.indices
    indptr = csgraph.indptr


    NE = data.size # number of edges

    rank = np.zeros(N, dtype=ITYPE)
    predecessors = np.arange(N, dtype=ITYPE) # this array is used to keep track of which tree each vertex belongs to

    data_dtype = data.dtype
    ITYPE_dtype = np.dtype(ITYPE)
    #create description of your table
    class Table_Description(tables.IsDescription):
        data = tables.Col.from_dtype(data_dtype, pos=0)
        col_ind = tables.Col.from_dtype(ITYPE_dtype, pos=1)
        row_ind = tables.Col.from_dtype(ITYPE_dtype, pos=2)

    #create hdf5 file and table
    if table_file == None:
        f = tables.open_file(os.path.join(arrays_dir, 'graph.hdf'),mode="w")
    else:
        f = table_file
    a = f.create_table("/","graph",
                       description=Table_Description,
                       expectedrows=NE)

    cs = a.chunkshape[0] # chunk size
    
    # fill table with data, col_ind and expanded indptr
    save_arrays_to_disk(data, indices, indptr, a)

    #Create a full index (on disk if you use the tmp_dir parameter
    a.cols.data.create_index(9, kind='full', tmp_dir=index_dir)    

    mst_n_edges = 0
    mst_data = np.zeros(N-1, dtype = data.dtype)
    mst_col_ind = np.empty(N-1, dtype = ITYPE)
    mst_row_ind = np.empty(N-1, dtype = ITYPE)

    n_chunks = NE // cs
    n_chunks_rest = NE % cs
    n_iters = 0

    read_gen = read_sorted_from_disk(a)
    for d, r, c in read_gen:
        # d = data, r = row_ind, c = col_ind
        iters, mst_n_edges = _min_spanning_tree_disk(d, c, r,
                                                   predecessors, rank,
                                                   mst_n_edges, mst_data,
                                                   mst_col_ind, mst_row_ind)
        n_iters += iters
        if mst_n_edges >= N-1:
            break

    f.close()

    # build the final MST matrix
    mst_data = mst_data[:mst_n_edges]
    mst_col_ind = mst_col_ind[:mst_n_edges]
    mst_row_ind = mst_row_ind[:mst_n_edges]
    sp_tree = csr_matrix((mst_data,(mst_col_ind,mst_row_ind)),
                         shape=(N,N))
    sp_tree.eliminate_zeros()

    return sp_tree

def minimum_spanning_tree_disk_file(file, index_dir='/tmp/'):
    global NULL_IDX

    with tables.openFile(file, 'a') as f:
        a = f.root.graph

        N = a._v_attrs.N
        NE = a._v_attrs.NE

        data_dtype = a.cols.data.dtype
        ITYPE_dtype = a.cols.col_ind.dtype

        rank = np.zeros(N, dtype=ITYPE_dtype)
        predecessors = np.arange(N, dtype=ITYPE_dtype) # this array is used to keep track of which tree each vertex belongs to

        cs = a.chunkshape[0] # chunk size
        
        #Create a full index (on disk if you use the tmp_dir parameter
        a.cols.data.create_index(9, kind='full', tmp_dir=index_dir)    

        mst_n_edges = 0
        mst_data = np.zeros(N-1, dtype = data_dtype)
        mst_col_ind = np.empty(N-1, dtype = ITYPE_dtype)
        mst_row_ind = np.empty(N-1, dtype = ITYPE_dtype)

        n_chunks = NE // cs
        n_chunks_rest = NE % cs
        n_iters = 0

        read_gen = read_sorted_from_disk(a)
        for d, r, c in read_gen:
            # d = data, r = row_ind, c = col_ind
            iters, mst_n_edges = _min_spanning_tree_disk(d, c, r,
                                                       predecessors, rank,
                                                       mst_n_edges, mst_data,
                                                       mst_col_ind, mst_row_ind)
            n_iters += iters
            if mst_n_edges >= N-1:
                break

    # build the final MST matrix
    mst_data = mst_data[:mst_n_edges]
    mst_col_ind = mst_col_ind[:mst_n_edges]
    mst_row_ind = mst_row_ind[:mst_n_edges]
    sp_tree = csr_matrix((mst_data,(mst_col_ind,mst_row_ind)),
                         shape=(N,N))
    sp_tree.eliminate_zeros()

    return sp_tree

def read_sorted_from_disk(table):
    """Generator to read the data, col_ind and row_ind arrays by chunks from
    disk. table is a PyTables table. The generator will return a tuple with a
    chunk from each array (data, row_ind, col_ind).
    """
    read_width = table.chunkshape[0]
    n_rows = table.nrows
    n_chunks = n_rows // read_width
    n_chunks_rest = n_rows % read_width

    # read n_chunks * read_width rows
    for i in xrange(n_chunks):
        start = i * read_width
        end = start + read_width
        coassoc_arrays = table.read_sorted("data", checkCSI=True, 
                                       start=start, stop=end)
        data = np.ascontiguousarray(coassoc_arrays['data'])
        row_ind = np.ascontiguousarray(coassoc_arrays['row_ind'])
        col_ind = np.ascontiguousarray(coassoc_arrays['col_ind'])
        yield (data, row_ind, col_ind)

    # read the remaining rows if any
    if end < n_rows:
        start = end
        coassoc_arrays = table.read_sorted("data", checkCSI=True, 
                                       start=start, stop=n_rows)        
        data = np.ascontiguousarray(coassoc_arrays['data'])
        row_ind = np.ascontiguousarray(coassoc_arrays['row_ind'])
        col_ind = np.ascontiguousarray(coassoc_arrays['col_ind'])
        yield (data, row_ind, col_ind)

def save_arrays_to_disk(data, col_ind, indptr, table):
    N = indptr.size - 1
    row_ind_dtype = col_ind.dtype
    table._v_attrs.N = N
    table._v_attrs.NE = data.size
    for row in xrange(N):
        start = indptr[row]
        end = indptr[row+1]
        table.append((data[start:end],
                      col_ind[start:end],
                      np.full(end-start, row, dtype=row_ind_dtype)))
@nb.njit
def _min_spanning_tree_disk(data,
                       col_indices,
                       row_indices,
                       predecessors,
                       rank,
                       n_edges_in_mst,
                       mst_data,
                       mst_row_indices,
                       mst_col_indices):
    # Work-horse routine for computing minimum spanning tree using
    #  Kruskal's algorithm.  By separating this code here, we get more
    #  efficient indexing.

    # i - the i to start in the current iteration (edges are processed in chunks)
    # n_edges - total number of edges 

    # cdef unsigned int i, j, V1, V2, R1, R2, n_edges_in_mst, n_verts
    # cdef DTYPE_t E
    n_verts = predecessors.shape[0]
    

    # step through the edges from smallest to largest.
    #  V1 and V2 are the vertices, and E is the edge weight connecting them.
    i = 0
    while i < data.shape[0] and n_edges_in_mst < n_verts - 1:
        V1 = row_indices[i]
        V2 = col_indices[i]
        E = data[i]

        mst_v1 = V1
        mst_v2 = V2

        # progress upward to the head node of each subtree
        R1 = V1
        while predecessors[R1] != R1:
            R1 = predecessors[R1]
        R2 = V2
        while predecessors[R2] != R2:
            R2 = predecessors[R2]

        # Compress both paths.
        while predecessors[V1] != R1:
            predecessors[V1] = R1
        while predecessors[V2] != R2:
            predecessors[V2] = R2
            
        # if the subtrees are different, then we connect them and keep the
        # edge.  Otherwise, we remove the edge: it duplicates one already
        # in the spanning tree.
        if R1 != R2:
            # add edge to mst forest 
            mst_data[n_edges_in_mst] = E
            mst_row_indices[n_edges_in_mst] = mst_v1            
            mst_col_indices[n_edges_in_mst] = mst_v2
            n_edges_in_mst += 1
            
            # Use approximate (because of path-compression) rank to try
            # to keep balanced trees.
            if rank[R1] > rank[R2]:
                predecessors[R2] = R1
            elif rank[R1] < rank[R2]:
                predecessors[R1] = R2
            else:
                predecessors[R2] = R1
                rank[R1] += 1
        
        i += 1

    iters = i

    return iters, n_edges_in_mst

def gen_indptr(ne, n, p=0.0005):
    indptr = np.empty(n+1, dtype=np.int32)
    indptr[0] = 0
    for i in range(1,n+1):
        rand = np.random.randint(0,int(p*ne),1)
        if indptr[i-1] == ne:
            indptr[i] == ne
        if indptr[i-1] + rand > ne:
            rand = ne - indptr[i-1]
        indptr[i] = indptr[i-1] + rand
    indptr[-1] = ne
    return indptr


if __name__ == '__main__':
    folder = "/home/diogoaos/QCThesis/datasets/gauss10e6_overlap/"
    name = "csr_100k"
    # data = np.load(folder + name + "_data.npy")
    # indices = np.load(folder + name + "_indices.npy")
    # indptr = np.load(folder + name + "_indptr.npy")
    # degree = indptr[1:] - indptr[:-1]

    n = 100000
    ne = n * 10
    data = np.random.randint(0, 255, ne).astype(np.uint8)
    indices = np.random.randint(0, n, ne).astype(np.int32)
    indptr = gen_indptr(n, ne)

    # convert to diassoc if necessary
    n_min = (data == data.min()).sum()
    n_max = (data == data.max()).sum()

    if n_min > n_max:
        data = data + 1 - data.max()

    class simple_csr:
        shape = None
        data = None
        indices = None
        indptr = None

    egCSR = simple_csr()
    egCSR.shape = (indptr.size-1,indptr.size-1)
    egCSR.data = data
    egCSR.indices = indices
    egCSR.indptr = indptr

    mst,i = minimum_spanning_tree_disk(egCSR)
    print i
    print mst.nnz
    print mst.data.size