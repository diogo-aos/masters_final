"""
@author: Diogo Silva

Various functions to build graphs.
"""

import numpy as np
from numba import njit, jit, cuda, int32, float32
from mst import memSet, compute_cuda_grid_dim
from MyML.utils.scan import scan_gpu as ex_prefix_sum_gpu,\
                            exprefixsumNumbaSingle as ex_prefix_sum_cpu,\
                            exprefixsumNumba as ex_prefix_sum_cpu2

#
# BORUVKA MST FUNCTIONS FOR BUILDING MST GRAPH
#

@jit
def binaryOriginVertexSearch(key, dest, fe, od):
    """
    Inputs:
        key         : edge id
        dest        : destination array where the i-th element
                      is the ID of the destination vertex of the
                      i-th edge
        fe          : first_edge array
        od          : outdegree array
    """
    imin = 0
    imax = fe.size

    while imin < imax:
        imid = (imax + imin) / 2

        imid_fe = fe[imid]
        # key is before
        if key < imid_fe:
            imax = imid
        # key is after
        elif key > imid_fe + od[imid] - 1:
            imin = imid + 1
        # key is between first edge of imid and next first edge
        else:
            return imid
    return -1

# @jit(["int32(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])",
#       "int32(int32[:], float32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], float32[:])"], nopython=True)
@jit(nopython=True)
def getGraphFromEdges_seq(dest, weight, fe, od, edges, nod, nfe, ndest, nweight):

    # first build the outDegree to get the first_edge
    for e in range(edges.size):
        edge = edges[e]
        o_v = dest[edge] # destination
        i_v = binaryOriginVertexSearch(edge, dest, fe, od)
        if i_v == -1:
            return -1
        nod[o_v] += 1
        nod[i_v] += 1

    # get first edge from outDegree
    ex_prefix_sum_cpu2(nod, nfe, init = 0)

    #get copy of newFirstEdge to serve as pointers for the newDest
    top_edge = np.empty(nfe.size, dtype = np.int32)
    for i in range(nfe.size):
        top_edge[i] = nfe[i]
    #top_edge = nfe.copy()

    # go through all the mst edges again and write the new edges in the new arrays
    for e in range(edges.size):
        edge = edges[e]

        o_v = dest[edge] # destination vertex
        i_v = binaryOriginVertexSearch(edge, dest, fe, od)
        if i_v == -1:
            return -1
        
        i_ptr = top_edge[i_v]
        o_ptr = top_edge[o_v]

        ndest[i_ptr] = o_v
        ndest[o_ptr] = i_v

        edge_w = weight[edge]
        nweight[i_ptr] = edge_w
        nweight[o_ptr] = edge_w

        top_edge[i_v] += 1
        top_edge[o_v] += 1

    return 0


def getGraphFromEdges_gpu(dest, weight, fe, od, edges, n_edges = None,
                          MAX_TPB = 512, stream = None):
    """
    All input (except MAX_TPB and stream) are device arrays.
    edges       : array with the IDs of the edges that will be part of the new graph
    n_edges     : array of 1 element with the number of valid edges in the edges array;
                  if n_edges < size of edges, the last elements of the edges array are
                  not considered
    """

    # check if number of valid edges was received
    if n_edges is None:
        edges_size = edges.size
        n_edges = cuda.to_device(np.array([edges_size], dtype = np.int32))
    else:
        edges_size = int(n_edges.getitem(0))

    # check if a stream was received, if not create one
    if stream is None:
        myStream = cuda.stream()
    else:
        myStream = stream
    
    new_n_edges = edges_size * 2

    # allocate memory for new graph
    ndest = cuda.device_array(new_n_edges, dtype = dest.dtype,
                              stream = myStream)
    nweight = cuda.device_array(new_n_edges, dtype = weight.dtype,
                                stream = myStream)
    nfe = cuda.device_array_like(fe, stream = myStream)
    nod = cuda.device_array_like(od, stream = myStream)

    # fill new outdegree with zeros
    vertexGrid = compute_cuda_grid_dim(nod.size, MAX_TPB)
    memSet[vertexGrid, MAX_TPB, myStream](nod, 0)

    # count all edges of new array and who they belong to
    edgeGrid = compute_cuda_grid_dim(edges_size, MAX_TPB)
    countEdges[edgeGrid, MAX_TPB, myStream](edges, n_edges, dest, fe, od, nod)

    # get new first_edge array from new outdegree
    nfe.copy_to_device(nod, stream=myStream)
    ex_prefix_sum_gpu(nfe, MAX_TPB = MAX_TPB, stream = myStream)


    # copy new first_edge to top_edge to serve as pointer in adding edges
    top_edge = cuda.device_array_like(nfe, stream = myStream)
    top_edge.copy_to_device(nfe, stream = myStream)

    addEdges[edgeGrid, MAX_TPB, myStream](edges, n_edges, dest, weight, fe, od,
                                          top_edge, ndest, nweight)

    del top_edge
    #del dest, weight, fe, od
    return ndest, nweight, nfe, nod



@cuda.jit
def countEdges(edges, n_edges, dest, fe, od, nod):
    # n_edges_sm = cuda.shared.array(0, dtype = int32)

    edge = cuda.grid(1)

    # if edge == 0:
    #     n_edges_sm[0] = n_edges[0]

    # if edge >= n_edges_sm[0]:
    if edge >= n_edges[0]:
        return

    key = edges[edge]

    # if edge is -1 it was marked for removal
    if key == -1:
        return

    o_v = dest[key]
    i_v = binaryOriginVertexSearch_CUDA(key, dest, fe, od)

    # increment edges on origin and destination vertices
    cuda.atomic.add(nod, i_v, 1)
    cuda.atomic.add(nod, o_v, 1)

@cuda.jit
def addEdges(edges, n_edges, dest, weight, fe, od, top_edge, ndest, nweight):
    n_edges_sm = cuda.shared.array(0, dtype = int32)

    edge = cuda.grid(1)

    # if edge == 0:
    #     n_edges_sm[0] = n_edges[0]

    key = edges[edge]

    # if edge is -1 it was marked for removal
    if key == -1:
        return

    o_v = dest[key]
    i_v = binaryOriginVertexSearch_CUDA(key, dest, fe, od)

    # get and increment pointers for each vertex
    i_ptr = cuda.atomic.add(top_edge, i_v, 1)
    o_ptr =cuda.atomic.add(top_edge, o_v, 1)

    # add edges to destination array
    ndest[i_ptr] = o_v
    ndest[o_ptr] = i_v

    # add weight to edges
    edge_w = weight[key]
    nweight[i_ptr] = edge_w
    nweight[o_ptr] = edge_w    



@cuda.jit(device=True)
def binaryOriginVertexSearch_CUDA(key, dest, fe, od):
    """
    TODO: test separately
    """
    imin = 0
    imax = fe.size

    while imin < imax:
        imid = (imax + imin) / 2

        imid_fe = fe[imid]
        # key is before
        if key < imid_fe:
            imax = imid
        # key is after
        elif key > imid_fe + od[imid] - 1:
            imin = imid + 1
        # key is between first edge of imid and next first edge
        else:
            return imid
    return -1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       FUNCTION TO BUILD MST GRAPH FROM SCIPY MST LEAN
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
@njit
def build_mst_graph(data, indices, mst_idx, mst_rows,
                    mst_data, mst_indices):
    """This function will build a new CSR graph (data, indices and indptr
    arrays) from an original graph and a selection of the edges that constitute
    its MST.
    The data and indices arrays are from the original graph.
    The mst_idx and mst_rows have the selection of the edges in the original
    graph and their originating edge (row).
    The mst_data and mst_indices are the arrays that will store the final graph.
    """

    n = mst_idx.size

    # use quicksort to have the edge to add in an increasing order relative to
    # to their row; facilitates building indptr
    quicksort_two(mst_rows, mst_idx, n)

    # mst_rows will be the degree; this is possible because the edges are
    # inserted in an increasing order row wise, the edge indices are always
    # bigger than the row to increment

    for i in range(n):
        edge_idx = mst_idx[i]
        edge_row = mst_rows[i]
        mst_data[i] = data[edge_idx]
        mst_indices[i] = indices[edge_idx]
        mst_rows[edge_row] += 1

    # exclusive prefix sum to convert from degree to indptr
    ex_prefix_sum_cpu(mst_rows, init = 0)


@njit
def outdegree_from_firstedge(firstedge, outdegree, n_edges):
    n_vertices = firstedge.size
    for v in range(n_vertices - 1):
        outdegree[v] = firstedge[v + 1] - firstedge[v]
    outdegree[n_vertices - 1] = n_edges - firstedge[n_vertices - 1]