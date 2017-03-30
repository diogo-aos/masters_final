import numpy as np
from numba import cuda, jit, void, int32, float32

from mst import boruvka_minho_seq,
                boruvka_minho_gpu,
                compute_cuda_grid_dim
from connected_components import connected_comps_seq,
                                 connected_comps_gpu
from build import getGraphFromEdges_gpu, getGraphFromEdges_seq

from numbapro.cudalib.sorting import RadixSort
from numbapro.cudalib.cublas import Blas

def sl_mst_lifetime_seq(dest, weight, fe, od, disconnect_weight = None):

    if disconnect_weight is None:
        disconnect_weight = weight.max()

    mst, n_edges = boruvka_minho_seq(dest, weight, fe, od)

    # Get array with only the considered weights in the MST
    # and remove those edges in the MST edge list
    mst_weights = weight[mst[:n_edges]]

    # Sort the MST weights. There are no repeated edges at this
    # point since the output MST is like a directed graph.
    sortedWeightArgs = mst_weights.argsort()
    mst_weights = mst_weights[sortedWeightArgs]
    mst = mst[sortedWeightArgs]

    # Allocate array for the lifetimes.
    lifetimes = mst_weights[1:] - mst_weights[:-1]

    arg_max_lt = lifetimes.argmax()
    max_lt = lifetimes[arg_max_lt]

    # this is the lifetime between edges with no connection and the weakest link
    #lt_threshold = disconnect_weight - max_lt
    lt_threshold = disconnect_weight - mst_weights[-1]

    # if the maximum lifetime if higher or equal than the lifetime threshold
    # cut the tree
    if max_lt >= lt_threshold:
        # from arg_max_lt onward all edges are discarded
        n_discarded = lifetimes.size - arg_max_lt + 1
        
        # remove edges
        mst = mst[:-n_discarded]

    del lifetimes, mst_weights

    ndest = np.empty(mst.size * 2, dtype = dest.dtype)
    nweight = np.empty(mst.size * 2, dtype = weight.dtype)
    nfe = np.empty_like(fe)
    nod = np.zeros_like(od)

    # build graph from mst
    getGraphFromEdges_seq(dest, weight, fe, od, mst,
                          nod, nfe, ndest, nweight)

    labels = connected_comps_seq(ndest, nweight, nfe, nod)

    del ndest, nweight, nfe, nod

    return labels



def sl_mst_lifetime_gpu(dest, weight, fe, od, disconnect_weight = None,
                        MAX_TPB = 256, stream = None):
    """
    Input are device arrays.
    Inputs:
     dest, weight, fe 		: device arrays
     disconnect_weight 		: weight between unconnected vertices
     mst 					: list of edges in MST
     MAX_TPB 				: number of threads per block
     stream 				: CUDA stream to use
    TODO:
     - argmax is from cuBlas and only works with 32/64 floats. Make this work 
       with any type.
     - 
    """

    if disconnect_weight is None:
        disconnect_weight = weight.max()

    if stream is None:
        myStream = cuda.stream()
    else:
        myStream = stream

    mst, n_edges = boruvka_minho_gpu(dest, weight, fe, od,
                                     MAX_TPB=MAX_TPB, stream=myStream,
    	  							 returnDevAry=True)

    # Allocate array for the mst weights.
    h_n_edges = int(n_edges.getitem(0, stream=myStream)) # edges to keep in MST
    mst_weights = cuda.device_array(h_n_edges, dtype=weight.dtype)    

    # Get array with only the considered weights in the MST
    # and remove those edges in the MST edge list
    mstGrid = compute_cuda_grid_dim(h_n_edges, MAX_TPB)
    d_weight = cuda.to_device(weight, stream = myStream)
    getWeightsOfEdges_gpu[mstGrid, MAX_TPB, myStream](mst, n_edges, d_weight,
                                                      mst_weights)    

    # Sort the MST weights. There are no repeated edges at this
    # point since the output MST is like a directed graph.
    sorter = RadixSort(maxcount = mst_weights.size, dtype = mst_weights.dtype,
                       stream = myStream)
    sortedWeightArgs = sorter.argsort(mst_weights)

    # Allocate array for the lifetimes.
    lifetimes = cuda.device_array(mst_weights.size - 1, dtype=mst_weights.dtype)
    compute_lifetimes_CUDA[mstGrid, MAX_TPB, myStream](mst_weights, lifetimes)

    maxer = Blas(stream)
    arg_max_lt = maxer.amax(lifetimes)
    max_lt = lifetimes.getitem(arg_max_lt)

    # this is the lifetime between edges with no connection and the weakest link
    #lt_threshold = disconnect_weight - max_lt
    lt_threshold = disconnect_weight - mst_weights.getitem(mst_weights.size - 1)

    # if the maximum lifetime is higher or equal than the lifetime threshold
    # cut the tree
    if max_lt >= lt_threshold:
        # from arg_max_lt onward all edges are discarded
        n_discarded = lifetimes.size - arg_max_lt + 1

        # remove edges
        removeGrid = compute_cuda_grid_dim(n_discarded, MAX_TPB)
        removeEdges[removeGrid, MAX_TPB](edgeList, sortedArgs, n_discarded)

        # compute new amount of edges and update it
        new_n_edges = h_n_edges - n_discarded
        cuda.to_device(np.array([new_n_edges], dtype = n_edges.dtype),
                       to = n_edges,
                       stream = myStream)

    ngraph = getGraphFromEdges_gpu(dest, weight, fe, od, edges = mst,
                                   n_edges = n_edges, MAX_TPB = MAX_TPB,
                                   stream = myStream)

    ndest, nweight, nfe, nod = ngraph

    labels = connected_comps_gpu(ndest, nweight, nfe, nod,
                                 MAX_TPB = 512, stream = myStream)

    del ndest, nweight, nfe, nod, lifetimes

    return labels




@cuda.jit
def removeEdges(edgeList, sortedArgs, n_discarded):
    """
    inputs:
        edgeList         : list of edges
        sortedArgs         : argument list of the sorted weight list
        n_discarded     : number of edges to be discarded specified in sortedArgs

    Remove discarded edges form the edge list.
    Each edge discarded is replaced by -1.

    Discard edges specified by the last n_discarded arguments
    in the sortedArgs list.

    """

    tgid = cuda.grid(1)

    # one thread per edge that must be discarded
    # total number of edges to be discarded is the difference 
    # between the between the total number of edges and the 
    # number of edges to be considered + the number edges 
    # to be discarded

    if tgid >= n_discarded:
        return

    # remove not considered edges
    elif tgid < n_considered_edges:
        maxIdx = edgeList.size - 1 # maximum index of sortedArgs
        index = maxIdx - tgid # index of 
        edgeList[index] = -1




@cuda.jit
def argmax_lvl0(ary, reduce_max, reduce_arg):
    """
    This only works for positive values arrays.
    Shared memory must be initialized with double the size of 
    the block size.
    """
    sm_ary = cuda.shared.array(shape = 0, dtype = ary.dtype)

    # each thread will process two elements
    tgid = cuda.grid(1)
    thid = cuda.threadIdx.x

    # pointer to value and argument side of shared memory
    val_pointer = 0
    arg_pointer = sm_ary.size / 2    

    # when global thread id is bigger or equal than the ary size
    # it means that the block is incomplete; in this case we just
    # fill the rest of the block with -1 so it is smaller than all
    # other elements; this only works for positive arrays
    if tgid < ary.size:
        sm_ary[val_pointer + thid] = ary[tgid]
        sm_ary[arg_pointer + thid] = tgid
    else:
        sm_ary[val_pointer + thid] = 0
        sm_ary[arg_pointer + thid] = -1        


    cuda.syncthreads()

    s = cuda.blockDim.x / 2
    while s >0:
        index = 2 * s * thid

        if thid < s:
            # only change if the left element is smaller than the right one
            if sm_ary[val_pointer + thid] < sm_ary[val_pointer + thid + s]:
                sm_ary[val_pointer + thid] = sm_ary[val_pointer + thid + s]
                sm_ary[arg_pointer + index] = sm_ary[arg_pointer + index + s]

        cuda.syncthreads()

    if thid == 0:
        reduce_ary[cuda.blockIdx.x] = sm_ary[val_pointer]
        reduce_arg[cuda.blockIdx.x] = sm_ary[arg_pointer]

@cuda.jit
def argmax_lvl1(reduce_max, reduce_arg):
    pass

@cuda.jit
def search_argmin_val(ary, val):
    tgid = cuda.grid(1)
    if tgid >= ary.size:
        return

@cuda.reduce
def max_gpu(a,b):
    if a >= b:
        return a
    else:
        return b

@cuda.jit
def compute_lifetimes_CUDA(nweight, lifetimes):
    edge = cuda.grid(1)
    
    if edge >= lifetimes.size:
        return
    
    lifetimes[edge] = nweight[edge + 1] - nweight[edge]

@cuda.jit#("void(int32[:],int32[:],int32[:],int32[:])")
           # "void(int32[:],int32[:],float32[:],float32[:])"])
def getWeightsOfEdges_gpu(edges, n_edges, weights, nweights):
    """
    This function will take a list of edges (edges), the number of edges to 
    consider (n_edges, the weights of all the possible edges (weights) and the 
    array for the weights of the list of edges and put the weight of each edge 
    in the list of edges in the nweights, in the same position.

    The kernel will also discard not considered edges, i.e. edges whose 
    argument >= n_edges.
    Discarding an edge is done by replacing the edge by -1.
    """
    # n_edges_sm = cuda.shared.array(1, dtype = int32)
    edge = cuda.grid(1)

    if edge >= edges.size:
        return
    
    # if edge == 0:
    #     n_edges_sm[0] = n_edges[0]
    # cuda.syncthreads()
    
    
    # if edge >= n_edges_sm[0]:
    if edge >= n_edges[0]:
        edges[edge] = -1
    else:
        myEdgeID = edges[edge]
        nweights[edge] = weights[myEdgeID]