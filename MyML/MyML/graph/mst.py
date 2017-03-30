# -*- coding: utf-8 -*-
"""
author: Diogo Silva
notes: Boruvka implementation based on Sousa's "A Generic and Highly Efficient Parallel Variant of Boruvka ’s Algorithm"
"""


import numpy as np
from MyML.utils.scan import scan_gpu as ex_prefix_sum_gpu,\
                            exprefixsumNumbaSingle as ex_prefix_sum_cpu,\
                            exprefixsumNumba as ex_prefix_sum_cpu2
from numba import jit, njit, cuda, void, boolean, int8, uint8, int32, float32


def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []

    # set to keep track of unvisited vertices
    # useful for unconnected graphs
    unvisited_vertices=set(xrange(n_vertices)) - {0}

    # initialize with node 0:
    visited_vertices = [0]
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf

    start_of_mst = [0]

    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)

        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]

        # when min is inf, it means the graph is unconnected
        # we add a vertex from the unvisited set
        if X[new_edge[0], new_edge[1]] == np.inf:
            added_vertex = unvisited_vertices.pop()
            new_edge = [added_vertex, np.argmin(X[added_vertex])]
            visited_vertices.append(added_vertex)  # add poped vertex to visited
            num_visited += 1

            start_of_mst.append(num_visited)  # add start of new independent MST

        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])

        # remove vertex from unvisited
        unvisited_vertices.discard(new_edge[1])

        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges), np.array(start_of_mst)


def minimum_spanning_tree_csr(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    raise Exception("NOT IMPLEMENTED")

    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")

    n_vertices = X.shape[0]
    spanning_edges = []

    # set to keep track of unvisited vertices
    # useful for unconnected graphs
    unvisited_vertices = set(xrange(n_vertices)) - {0}

    # initialize with node 0:
    visited_vertices = [0]
    num_visited = 1
    # exclude self connections:
    # diag_indices = np.arange(n_vertices)
    # X[diag_indices, diag_indices] = np.inf

    start_of_mst = [0]
    mst_edges = set()

    while num_visited != n_vertices:
        # get shortest edge from visited vertices

        min_weight = np.inf

        for vertex in visited_vertices:

            # check if vertex has any edges
            np.where(X[vertex].data < min_weight)

        new_edge = np.argmin(X[visited_vertices], axis=None)

        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]

        # when min is inf, it means the graph is unconnected
        # we add a vertex from the unvisited set
        if X[new_edge[0],new_edge[1]] == np.inf:
            added_vertex = unvisited_vertices.pop()
            new_edge=[added_vertex,np.argmin(X[added_vertex])]
            visited_vertices.append(added_vertex) # add poped vertex to visited
            num_visited += 1

            start_of_mst.append(num_visited) # add start of new independent MST

        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])

        # remove vertex from unvisited
        unvisited_vertices.discard(new_edge[1])

        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges), np.array(start_of_mst)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#
#                      NUMBA CPU BORUVKA
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def boruvka_minho_seq(dest_in, weight_in, firstEdge_in, outDegree_in):
    dest = dest_in
    weight = weight_in
    firstEdge = firstEdge_in
    outDegree = outDegree_in

    n_vertices = firstEdge.size
    n_edges = dest.size

    edge_id = np.arange(n_edges, dtype=dest.dtype)

    n_components = n_vertices
    n_mst = 1

    # maximum size of MST is when it is connected
    mst = np.empty(n_vertices - 1, dtype=dest.dtype)
    mst_pointer = 0

    # top_edge is recycled between iterations
    top_edge = np.empty(n_components, dtype=dest.dtype)

    final_converged = False
    while(not final_converged):

        vertex_minedge = top_edge

        findMinEdgeNumba(vertex_minedge, weight, firstEdge, outDegree, dest)

        removeMirroredNumba(vertex_minedge, dest)

        # add new edges to final MST and update MST pointer
        mst_pointer = addEdgesToMSTNumba(mst, mst_pointer, vertex_minedge, edge_id)

        # intialize colors of current graph
        colors = np.empty(n_components, dtype=dest.dtype)
        initColorsNumba(vertex_minedge, dest, colors)

        # propagate colors until convergence
        converged = False
        while(not converged):
            converged = propagateColorsNumba(colors)

        # flag marks the vertices that are the representatives of the new supervertices
        # flag = np.where(vertex_minedge == -1, 1, 0).astype(np.int32) # get super-vertives representatives
        # del vertex_minedge # vertex_minedge no longer necessary for next steps

        new_vertex = vertex_minedge
        buildFlag(colors, new_vertex)

        new_n_vertices = ex_prefix_sum_cpu(new_vertex, init=0)

        if new_n_vertices == 1:
            final_converged = True
            break

        # count number of edges for new supervertices and write in new outDegree
        newOutDegree = np.zeros(new_n_vertices, dtype=dest.dtype)
        countNewEdgesNumba(colors, firstEdge, outDegree, dest, new_vertex, newOutDegree)

        # new first edge array for contracted graph
        newFirstEdge = np.empty(newOutDegree.size, dtype=dest.dtype)
        new_n_edges = ex_prefix_sum_cpu2(newOutDegree, newFirstEdge, init=0)

        # if no edges remain, then MST has converged
        if new_n_edges == 0:
            final_converged = True
            break

        # create arrays for new edges
        new_dest = np.empty(new_n_edges, dtype=dest.dtype)
        new_edge_id = np.empty(new_n_edges, dtype=dest.dtype)
        new_weight = np.empty(new_n_edges, dtype=weight.dtype)
        top_edge = newFirstEdge.copy()

        # assign and insert new edges
        assignInsertNumba(edge_id, dest, weight, firstEdge, outDegree, colors,
                          new_vertex, new_dest, new_edge_id, new_weight, top_edge)

        # delete old graph
        del new_vertex, edge_id, dest, weight, firstEdge, outDegree, colors

        # write new graph
        n_components = newFirstEdge.size
        edge_id = new_edge_id
        dest = new_dest
        weight = new_weight
        firstEdge = newFirstEdge
        outDegree = newOutDegree

    return mst, mst_pointer


@jit(["void(int32[:],int32[:])"], nopython=True)
def buildFlag(colors, flag):
    n_components = colors.size

    for v in range(n_components):
        if v == colors[v]:
            flag[v] = 1
        else:
            flag[v] = 0


# @jit(["void(int32[:],float32[:],int32[:],int32[:],int32[:])",
#       "void(int32[:],uint8[:],int32[:],int32[:],int32[:])",
#       "void(int32[:],int32[:],int32[:],int32[:],int32[:])"], nopython=True)
@jit(nopython=True)
def findMinEdgeNumba(vertex_minedge, weight, firstEdge, outDegree, dest):

    n_components = vertex_minedge.size

    for v in range(n_components):
        v_n_edges = outDegree[v]
        if v_n_edges == 0:
            vertex_minedge[v] = -1
            continue

        start = firstEdge[v]  # initial edge
        end = start + v_n_edges  # initial edge of next vertex

        min_edge = start
        min_weight = weight[start]
        min_dest = dest[start]

        # loop through all the edges of vertex to get the minimum
        for edge in range(start + 1, end):
            edge_weight = weight[edge]
            edge_dest_curr = dest[edge]
            if edge_weight < min_weight:
                min_edge = edge
                min_weight = edge_weight
                min_dest = edge_dest_curr
            elif edge_weight == min_weight and edge_dest_curr < min_dest:
                min_edge = edge
                min_weight = edge_weight
                min_dest = edge_dest_curr
        vertex_minedge[v] = min_edge


@jit(["void(int32[:],int32[:])"], nopython=True)
def removeMirroredNumba(vertex_minedge, dest):
    n_components = vertex_minedge.size
    for v in range(n_components):  # for each vertex

        myEdge = vertex_minedge[v]  # my edge

        if myEdge == -1:
            continue

        my_succ = dest[myEdge]  # my successor
        succ_edge = vertex_minedge[my_succ]  # my successor's edge

        # if my successor's edge is -1 it means it was already removed
        if succ_edge == -1:
            continue

        succ_succ = dest[succ_edge]  # my successor's successor

        # if my successor's successor is me then remove my edge if my ID
        # is lower that my successor's ID
        if v == succ_succ:
            if v < my_succ:
                vertex_minedge[v] = -1


@jit(["int32(int32[:],int32,int32[:],int32[:])"], nopython=True)
def addEdgesToMSTNumba(mst, mst_pointer, vertex_minedge, edge_id):
        n_components = vertex_minedge.size

        for v in range(n_components):
            my_edge = vertex_minedge[v]
            if my_edge != -1:
                mst[mst_pointer] = edge_id[my_edge]
                mst_pointer += 1
        return mst_pointer


@jit(["void(int32[:],int32[:],int32[:])"], nopython=True)
def initColorsNumba(vertex_minedge, dest, colors):

        n_components = vertex_minedge.size

        for v in range(n_components):
            my_edge = vertex_minedge[v]
            if my_edge != -1:
                colors[v] = dest[my_edge]
            else:
                colors[v] = v


@jit(["boolean(int32[:])"])
def propagateColorsNumba(colors):
    '''
    For checking convergence, start with a boolean variable converged set to True.
    At each assignment the new color is compared to the old one. The result
    of this comparison (True of False) is used to perform a boolean AND with converged.
    If all the new colors are equal to the old colors then the result of all the ANDs is
    True and convergence was met.
    '''
    n_components = colors.size
    converged = True
    for v in range(n_components):  # for each vertex
        my_color = colors[v]  # my_color is also my successor
        if my_color != v:
            new_color = colors[my_color]  # my new color is the color of my successor
            if new_color != my_color:
                converged = False
            colors[v] = new_color  # assign new color
    return converged


@jit(["void(int32[:],int32[:],int32[:],int32[:],int32[:],int32[:])"], nopython=True)
def countNewEdgesNumba(colors, firstEdge, outDegree, dest, new_vertex, newOutDegree):
    # new number of vertices is the number of representatives

    n_components = colors.size
    for v in range(n_components):
        my_color = colors[v]  # my color
        my_color_id = new_vertex[my_color]  # vertex id of my color (supervertex)

        # my edges
        startW = firstEdge[v]
        endW = startW + outDegree[v]

        for edge in range(startW, endW):
            my_succ = dest[edge]
            my_succ_color = colors[my_succ]

            if my_color != my_succ_color:
                newOutDegree[my_color_id] += 1  # increment number of outgoing edges of super-vertex


# @jit(["void(int32[:], int32[:], float32[:], int32[:], int32[:], int32[:], int32[:], int32[:],int32[:],float32[:],int32[:])",
#       "void(int32[:], int32[:], uint8[:], int32[:], int32[:], int32[:], int32[:], int32[:],int32[:],uint8[:],int32[:])",
#       "void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:],int32[:],int32[:],int32[:])"])
@njit
def assignInsertNumba(edge_id, dest, weight, firstEdge,
                      outDegree, colors, new_vertex,
                      new_dest, new_edge_id, new_weight, top_edge):

    n_components = colors.size

    for v in range(n_components):
        my_color = colors[v]  # my color

        # my edges
        startW = firstEdge[v]
        endW = startW + outDegree[v]

        for edge in range(startW, endW):
            my_succ = dest[edge]  # my successor
            my_succ_color = colors[my_succ]  # my successor's color

            # keep edge if colors are different
            if my_color != my_succ_color:
                supervertex_id = new_vertex[my_color]
                succ_supervertex_id = new_vertex[my_succ_color]  # supervertex id of my succ color

                pointer = top_edge[supervertex_id]  # where to add edge

                new_dest[pointer] = succ_supervertex_id
                new_weight[pointer] = weight[edge]
                new_edge_id[pointer] = edge_id[edge]

                top_edge[supervertex_id] += 1  # increment pointer of current supervertex


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#
#                      CUDA BORUVKA
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def boruvka_minho_gpu(dest_in, weight_in, firstEdge_in, outDegree_in,
                      MAX_TPB=512, stream=None, returnDevAry=False):

    n_vertices = firstEdge_in.size
    n_edges = dest_in.size
    n_components = n_vertices  # initial number of components is the number of vertices

    # get stream to use or create new one
    if stream is None:
        myStream = cuda.stream()
    else:
        myStream = stream

    ## check if input arrys are device arrays, if not transfer everything
    # to device
    if not isinstance(dest_in, cuda.cudadrv.devicearray.DeviceNDArray):
        dest = cuda.to_device(dest_in, stream=myStream)
    else:
        dest = dest_in

    if not isinstance(weight_in, cuda.cudadrv.devicearray.DeviceNDArray):
        weight = cuda.to_device(weight_in, stream=myStream)
    else:
        weight = weight_in

    if not isinstance(firstEdge_in, cuda.cudadrv.devicearray.DeviceNDArray):
        firstEdge = cuda.to_device(firstEdge_in, stream=myStream)
    else:
        firstEdge = firstEdge_in

    if not isinstance(outDegree_in, cuda.cudadrv.devicearray.DeviceNDArray):
        outDegree = cuda.to_device(outDegree_in, stream=myStream)
    else:
        outDegree = firstEdge_in

    # allocate array for edge IDs
    edge_id = cuda.to_device(np.arange(n_edges, dtype=np.int32), stream=myStream)

    # maximum size of MST is when it is connected = #vertices - 1
    mst = cuda.device_array(shape=n_vertices - 1, dtype=np.int32, stream=myStream)
    mst_pointer = cuda.to_device(np.zeros(1, dtype=np.int32), stream=myStream)

    # vertex_minedge reuses the top_edge array in the beginning of every iteration
    # which means we have to define it before the first iteration
    top_edge = cuda.device_array(n_components, dtype=np.int32, stream=myStream)

    converged = cuda.device_array(1, dtype=np.int8, stream=myStream)

    gridDim = compute_cuda_grid_dim(n_components, MAX_TPB)

    final_converged = False

    while(not final_converged):

        # assign vertex_minedge an array of size n_components
        # it eventually gets deleted in the end of the iteration
        # after it also served as the new_vertex
        vertex_minedge = top_edge

        findMinEdge_CUDA[gridDim, MAX_TPB, myStream](weight, firstEdge, outDegree, vertex_minedge, dest)

        removeMirroredEdges_CUDA[gridDim, MAX_TPB, myStream](dest, vertex_minedge)

        addEdgesToMST_CUDA[gridDim, MAX_TPB, myStream](mst, mst_pointer, vertex_minedge, edge_id)

        colors = cuda.device_array(shape=n_components, dtype=np.int32, stream=myStream)
        initializeColors_CUDA[gridDim, MAX_TPB, myStream](dest, vertex_minedge, colors)

        # propagate colors until convergence
        propagateConverged = False
        while(not propagateConverged):
            propagateColors_CUDA[gridDim, MAX_TPB, myStream](colors, converged)
            # if myStream is not 0:
            #     myStream.synchronize()
            converged_num = converged.getitem(0, stream=myStream)
            propagateConverged = True if converged_num == 1 else False

        # first we build the flags in the new_vertex array
        new_vertex = vertex_minedge  # reuse the vertex_minedge array as the new new_vertex
        buildFlag_CUDA[gridDim, MAX_TPB, myStream](colors, new_vertex)

        # new_n_vertices is the number of vertices of the new contracted graph
        new_n_vertices = ex_prefix_sum_gpu(new_vertex, MAX_TPB=MAX_TPB, stream=myStream).getitem(0, stream=myStream)
        new_n_vertices = int(new_n_vertices)

        if new_n_vertices == 1:
            final_converged = True
            del new_vertex
            break

        newGridDim = compute_cuda_grid_dim(new_n_vertices, MAX_TPB)
        
        # count number of edges for new supervertices and write in new outDegree
        newOutDegree = cuda.device_array(shape=new_n_vertices, dtype=np.int32, stream=myStream)
        memSet[newGridDim, MAX_TPB, myStream](newOutDegree, 0)  # zero the newOutDegree array
        countNewEdges_CUDA[gridDim, MAX_TPB, myStream](colors, firstEdge, outDegree, dest, new_vertex, newOutDegree)

        # new first edge array for contracted graph
        newFirstEdge = cuda.device_array_like(newOutDegree, stream=myStream)
        newFirstEdge.copy_to_device(newOutDegree, stream=myStream)  # copy newOutDegree to newFirstEdge
        new_n_edges = ex_prefix_sum_gpu(newFirstEdge, MAX_TPB=MAX_TPB, stream=myStream)

        new_n_edges = new_n_edges.getitem(0, stream=myStream)
        new_n_edges = int(new_n_edges)

        # if no edges remain, then MST has converged
        if new_n_edges == 0:
            final_converged = True
            del newOutDegree, newFirstEdge, new_vertex
            break

        # create arrays for new edges
        new_dest = cuda.device_array(new_n_edges, dtype=np.int32, stream=myStream)
        new_edge_id = cuda.device_array(new_n_edges, dtype=np.int32, stream=myStream)
        new_weight = cuda.device_array(new_n_edges, dtype=weight.dtype, stream=myStream)

        top_edge = cuda.device_array_like(newFirstEdge, stream=myStream)
        top_edge.copy_to_device(newFirstEdge, stream=myStream)

        # assign and insert new edges
        assignInsert_CUDA[gridDim, MAX_TPB, myStream](edge_id, dest, weight, firstEdge,
                                            outDegree, colors, new_vertex,
                                            new_dest, new_edge_id, new_weight, top_edge)

        # delete old graph
        del new_vertex, edge_id, dest, weight, firstEdge, outDegree, colors

        # write new graph
        n_components = newFirstEdge.size
        edge_id = new_edge_id
        dest = new_dest
        weight = new_weight
        firstEdge = newFirstEdge
        outDegree = newOutDegree
        gridDim = newGridDim

    del dest, weight, edge_id, firstEdge, outDegree, converged

    if returnDevAry:
        return mst, mst_pointer
    else:
        host_mst = mst.copy_to_host(stream=myStream)
        # no custom stream here to ensure synchronization
        mst_size = mst_pointer.getitem(0)

        del mst, mst_pointer

        return host_mst, mst_size


#@jit("int32(int32, int32)", nopython=True)
@jit(nopython=True)
def compute_cuda_grid_dim(n, tpb):
    bpg = np.ceil(n / np.float32(tpb))
    return np.int32(bpg)


@cuda.jit("void(int32[:])")
def initEdgeId(edgeId):
    e = cuda.grid(1)
    n_edges = edgeId.size
    if e >= n_edges:
        return

    edgeId[e] = e


@cuda.jit
def memSet(in_ary, val):
    idx = cuda.grid(1)
    if idx < in_ary.size:
        in_ary[idx] = val


#@cuda.jit(["void(float32[:], int32[:], int32[:], int32[:], int32[:])", "void(int32[:], int32[:], int32[:], int32[:], int32[:])"])
@cuda.jit
def findMinEdge_CUDA(weight, firstedge, outdegree, vertex_minedge, dest):
    v = cuda.grid(1)
    n_components = vertex_minedge.size

    # a thread per vertex
    if v >= n_components:
        return

    v_n_edges = outdegree[v]

    # isolated supervertex
    if v_n_edges == 0:
        vertex_minedge[v] = -1
        return

    start = firstedge[v]  # initial edge
    end = start + v_n_edges  # initial edge of next vertex

    min_edge = start
    min_weight = weight[start]  # get first weight for comparison inside loop
    min_dest = dest[start]

    # loop through all the edges of vertex to get the minimum
    for edge in range(start + 1, end):
        edge_weight = weight[edge]
        edge_dest_curr = dest[edge]
        if edge_weight < min_weight:
            min_edge = edge
            min_weight = edge_weight
            min_dest = edge_dest_curr
        elif edge_weight == min_weight and edge_dest_curr < min_dest:
            min_edge = edge
            min_weight = edge_weight
            min_dest = edge_dest_curr
    vertex_minedge[v] = min_edge


@cuda.jit("void(int32[:], int32[:])")
def removeMirroredEdges_CUDA(destination, vertex_minedge):
    v = cuda.grid(1)
    n_components = vertex_minedge.size

    # a thread per vertex
    if v >= n_components:
        return

    ########################
    my_edge = vertex_minedge[v]

    # removed vertex
    if my_edge == -1:
        return

    my_successor = destination[my_edge]
    successor_edge = vertex_minedge[my_successor]

    # successor already processed and its edge removed
    # because it was a mirrored edge with this vertex or another
    # either way nothing to do here
    if successor_edge == -1:
        return

    successor_successor = destination[successor_edge]

    # if the successor of the vertex's successor is the vertex itself AND
    # the vertex's ID is smaller than its successor, than remove its edge
    if v == successor_successor:
        if v < my_successor:
            vertex_minedge[v] = -1
        # else:
        #     vertex_minedge[my_successor] = -1


@cuda.jit("void(int32[:], int32[:], int32[:], int32[:])")
def addEdgesToMST_CUDA(mst, mst_pointer, vertex_minedge, edge_id):
    v = cuda.grid(1)
    n_components = vertex_minedge.size

    # a thread per vertex
    if v >= n_components:
        return

    my_edge = vertex_minedge[v]
    if my_edge != -1:
        pointer = cuda.atomic.add(mst_pointer, 0, 1)
        mst[pointer] = edge_id[my_edge]


@cuda.jit("void(int32[:], int32[:], int32[:])")
def initializeColors_CUDA(destination, vertex_minedge, colors):
    v = cuda.grid(1)
    n_components = vertex_minedge.size

    if v >= n_components:
        return

    ########################
    my_edge = vertex_minedge[v]

    if my_edge == -1:
        colors[v] = v
    else:
        my_successor = destination[my_edge]
        colors[v] = my_successor


@cuda.jit("void(int32[:], int8[:])")
def propagateColors_CUDA(colors, converged):
    #
    # Something bad (although very unlikely) can happen here. Consider the situation where only
    # one block (that is not the first one) doesn't converge. Consider also that that block
    # finished before the first threat of the first block reached the sentence converged[0] = 1.
    # Then, the kernel will report that there was convergence while, in fact, one block didn't converge.
    # This is very unlikely because it requires that the first threat of the first block takes more time reaching 
    # the 6th line of code that the time it takes a whole block to finish.
    # It is also very easy to fix my making the initialization of colors to be the kernel that sets
    # converged[0] = 1 -> not really because I have to set it to 1 every iteration of propagation
    #
    sm_converged = cuda.shared.array(shape=1, dtype=int8)

    v = cuda.grid(1)
    thid = cuda.threadIdx.x
    n_components = colors.size

    if v == 0:
        converged[0] = 1

    if thid == 0:
        sm_converged[0] = 1

    if v >= n_components:
        return

    cuda.syncthreads()

    my_color = colors[v]  # colour of vertex # n
    color_of_successor = colors[my_color]  # colour of successor of vertex

    # if my colour is different from that of my successor
    if my_color != color_of_successor:
        colors[v] = color_of_successor
        sm_converged[0] = 0

    if thid == 0:
        if sm_converged[0] == 0:
            converged[0] = 0


@cuda.jit("void(int32[:], int32[:])")
def buildFlag_CUDA(colors, flag):
    v = cuda.grid(1)
    n_components = colors.size

    if v >= n_components:
        return

    if v == colors[v]:
        flag[v] = 1
    else:
        flag[v] = 0


@cuda.jit("void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])")
def countNewEdges_CUDA(colors, firstEdge, outDegree, dest, new_vertex, newOutDegree):
    v = cuda.grid(1)  # vertex id is the global ID of thread
    n_components = colors.size

    if v >= n_components:
        return

    my_color = colors[v]
    my_color_id = new_vertex[my_color]

    startW = firstEdge[v]  # start of my edges
    endW = startW + outDegree[v]  # end of my edges

    for edge in range(startW, endW):
        my_succ = dest[edge]
        my_succ_color = colors[my_succ]

        if my_color != my_succ_color:
            cuda.atomic.add(newOutDegree, my_color_id, 1)


# @cuda.jit(["void(int32[:], int32[:], float32[:], int32[:],\
#                 int32[:], int32[:], int32[:], \
#                 int32[:], int32[:], float32[:], int32[:])",
#            "void(int32[:], int32[:], int32[:], int32[:],\
#                 int32[:], int32[:], int32[:], \
#                 int32[:], int32[:], int32[:], int32[:])"])
@cuda.jit
def assignInsert_CUDA(edge_id, dest, weight, firstEdge,
                      outDegree, colors, new_vertex,
                      new_dest, new_edge_id, new_weight, top_edge):
    v = cuda.grid(1)  # vertex id is the global ID of thread
    n_components = colors.size

    if v >= n_components:
        return

    my_color = colors[v]

    startW = firstEdge[v]
    endW = startW + outDegree[v]

    for edge in range(startW, endW):
        my_succ = dest[edge]
        my_succ_color = colors[my_succ]

        if my_color != my_succ_color:
            supervertex_id = new_vertex[my_color]
            succ_supervertex_id = new_vertex[my_succ_color]
            pointer = cuda.atomic.add(top_edge, supervertex_id, 1)  # get pointer and increment

            new_dest[pointer] = succ_supervertex_id
            new_weight[pointer] = weight[edge]
            new_edge_id[pointer] = edge_id[edge]
