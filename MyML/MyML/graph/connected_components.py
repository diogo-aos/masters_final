# -*- coding: utf-8 -*-
"""
author: Diogo Silva
notes: Boruvka implementation based on Sousa's "A Generic and Highly Efficient
       Parallel Variant of Boruvka ’s Algorithm"
       connected components from Boruvka
"""


import numpy as np
from MyML.utils.scan import exprefixsumNumbaSingle as ex_prefix_sum_cpu
from MyML.utils.scan import exprefixsumNumba as ex_prefix_sum_cpu2
from MyML.utils.scan import scan_gpu as ex_prefix_sum_gpu

from mst import findMinEdgeNumba
from mst import removeMirroredNumba
from mst import initColorsNumba
from mst import propagateColorsNumba
from mst import buildFlag
from mst import countNewEdgesNumba
from mst import assignInsertNumba
from mst import compute_cuda_grid_dim
from mst import findMinEdge_CUDA
from mst import removeMirroredEdges_CUDA
from mst import memSet
from mst import initializeColors_CUDA
from mst import propagateColors_CUDA
from mst import buildFlag_CUDA
from mst import countNewEdges_CUDA
from mst import assignInsert_CUDA

from numba import jit, cuda, void, boolean, int8, int32, float32

def connected_comps_mst_seq(mst, dest, fe):
    pass


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#
#                      NUMBA CPU 
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def connected_comps_seq(dest_in, weight_in, firstEdge_in, outDegree_in):
    dest = dest_in
    weight = weight_in
    firstEdge = firstEdge_in
    outDegree = outDegree_in

    n_vertices = firstEdge.size
    n_edges = dest.size

    n_components = n_vertices

    # still need edge_id for conflict resolution in find_minedge
    edge_id = np.arange(n_edges, dtype = dest.dtype)
    
    #labels = np.empty(n_vertices, dtype = dest.dtype)
    first_iter = True

    #initialize with name top_edge so we can recycle an array between iterations
    top_edge = np.empty(n_components, dtype = dest.dtype)

    final_converged = False
    while(not final_converged):
        vertex_minedge = top_edge
        findMinEdgeNumba(vertex_minedge, weight, firstEdge, outDegree, dest)
        removeMirroredNumba(vertex_minedge, dest)

        # intialize colors of current graph
        colors = np.empty(n_components, dtype = dest.dtype)
        initColorsNumba(vertex_minedge, dest, colors)

        # propagate colors until convergence
        converged = False
        while(not converged):
            converged = propagateColorsNumba(colors)

        # flag marks the vertices that are the representatives of the 
        # new supervertices new_vertex will be initialized with he flags
        new_vertex = vertex_minedge
        buildFlag(colors, new_vertex)
    
        new_n_vertices = ex_prefix_sum_cpu(new_vertex, init = 0)

        if first_iter:
            # first iteration defines labels as the initial colors and updates
            labels = colors.copy()
            first_iter = False

        # update the labels with the new colors
        update_labels_single_pass(labels, colors, new_vertex)        


        if new_n_vertices == 1:
            final_converged = True
            break        

        # count number of edges for new supervertices and write in new outDegree
        newOutDegree = np.zeros(new_n_vertices, dtype = dest.dtype)
        countNewEdgesNumba(colors, firstEdge, outDegree, dest,
                           new_vertex, newOutDegree)

        # new first edge array for contracted graph
        newFirstEdge = np.empty(newOutDegree.size, dtype = dest.dtype)
        new_n_edges = ex_prefix_sum_cpu2(newOutDegree, newFirstEdge, init = 0)

        # if no edges remain, then MST has converged
        if new_n_edges == 0:
            final_converged = True
            break

        # create arrays for new edges
        new_dest = np.empty(new_n_edges, dtype = dest.dtype)
        new_edge_id = np.empty(new_n_edges, dtype = dest.dtype)
        new_weight = np.empty(new_n_edges, dtype = weight.dtype)
        top_edge = newFirstEdge.copy()

        # assign and insert new edges
        assignInsertNumba(edge_id, dest, weight, firstEdge,
                          outDegree, colors, new_vertex, new_dest,
                          new_edge_id, new_weight, top_edge)

        # delete old graph
        del new_vertex, edge_id, dest, weight, firstEdge, outDegree, colors

        # write new graph
        n_components = newFirstEdge.size
        edge_id = new_edge_id
        dest = new_dest
        weight = new_weight
        firstEdge = newFirstEdge
        outDegree = newOutDegree

    return labels


@jit
def update_labels_numba(labels, update_array):
    """
    This kernel is dual purpose.

    This kernel updates the color of each vertex in the graph.
    The current colors of the vertices are indices for the current components 
    in the graph. Each current component in the graph will have a new color.
    The new color of vertex v will be the new color colors[curr_color] of its
    color labels[v] (index of a component). This step happens because the new
    color propagation is representative of the contracted graph. In this phase
    the update_array is the colors array.

    Old components are merged into new ones with new IDs. The labels need to
    have the new IDs. In this phase the update_array is the new_vertex array.
    """
    n_components = labels.size

    for v in range(n_components):
        curr_color = labels[v]
        labels[v] = update_array[curr_color]

@jit
def update_labels_single_pass(labels, colors, new_vertex):
    """
    Does all the updates on a single pass
    """
    n_components = labels.size

    for v in range(n_components):
        curr_color = labels[v]
        new_color = colors[curr_color]
        new_color_id = new_vertex[new_color]
        labels[v] = new_color_id


def connected_comps_gpu(dest_in, weight_in, firstEdge_in, outDegree_in,
                        MAX_TPB = 512, stream = None):
    if stream is None:
        myStream = cuda.stream()
    else:
        myStream = stream

    dest = cuda.to_device(dest_in, stream = myStream)
    weight = cuda.to_device(weight_in, stream = myStream)
    firstEdge = cuda.to_device(firstEdge_in, stream = myStream)
    outDegree = cuda.to_device(outDegree_in, stream = myStream)

    n_vertices = firstEdge.size
    n_edges = dest.size

    n_components = n_vertices

    # still need edge_id for conflict resolution in find_minedge
    edge_id = cuda.to_device(np.arange(n_edges, dtype = dest.dtype),
                                       stream = myStream)
    
    #labels = np.empty(n_vertices, dtype = dest.dtype)
    first_iter = True

    # initialize with name top_edge so we can recycle an array between iterations
    top_edge = cuda.device_array(n_components, dtype = dest.dtype,
                                 stream = myStream)
    labels = cuda.device_array(n_components, dtype = dest.dtype,
                               stream = myStream)

    converged = cuda.device_array(1, dtype = np.int8, stream = myStream)
    
    gridDimLabels = compute_cuda_grid_dim(n_components, MAX_TPB)
    gridDim = compute_cuda_grid_dim(n_components, MAX_TPB)

    final_converged = False
    while(not final_converged):
        vertex_minedge = top_edge

        findMinEdge_CUDA[gridDim, MAX_TPB, myStream](weight, firstEdge,
                                                     outDegree, vertex_minedge,
                                                     dest)

        removeMirroredEdges_CUDA[gridDim, MAX_TPB, myStream](dest, vertex_minedge)

        colors = cuda.device_array(shape = n_components, dtype = np.int32,
                                   stream = myStream)
        initializeColors_CUDA[gridDim, MAX_TPB, myStream](dest, vertex_minedge,
                                                          colors)

        # propagate colors until convergence
        propagateConverged = False
        while(not propagateConverged):
            propagateColors_CUDA[gridDim, MAX_TPB, myStream](colors, converged)
            converged_num = converged.getitem(0, stream = myStream)
            propagateConverged = True if converged_num == 1 else False

        # first we build the flags in the new_vertex array
        new_vertex = vertex_minedge # reuse the vertex_minedge array as the new new_vertex
        buildFlag_CUDA[gridDim, MAX_TPB, myStream](colors, new_vertex)

        # new_n_vertices is the number of vertices of the new contracted graph
        new_n_vertices = ex_prefix_sum_gpu(new_vertex, MAX_TPB = MAX_TPB, stream = myStream).getitem(0, stream = myStream)
        new_n_vertices = int(new_n_vertices)

        if first_iter:
            # first iteration defines labels as the initial colors and updates
            labels.copy_to_device(colors, stream = myStream)
            first_iter = False
        
        # other iterations update the labels with the new colors
        update_labels_single_pass_cuda[gridDimLabels, MAX_TPB, myStream](labels, colors, new_vertex)

        if new_n_vertices == 1:
            final_converged = True
            del new_vertex
            break

        newGridDim = compute_cuda_grid_dim(n_components, MAX_TPB)
        
        # count number of edges for new supervertices and write in new outDegree
        newOutDegree = cuda.device_array(shape = new_n_vertices,
                                         dtype = np.int32,
                                         stream = myStream)
        memSet[newGridDim, MAX_TPB, myStream](newOutDegree, 0) # zero the newOutDegree array
        countNewEdges_CUDA[gridDim, MAX_TPB, myStream](colors, firstEdge,
                                                       outDegree, dest,
                                                       new_vertex, newOutDegree)

        # new first edge array for contracted graph
        newFirstEdge = cuda.device_array_like(newOutDegree, stream = myStream)
        # copy newOutDegree to newFirstEdge
        newFirstEdge.copy_to_device(newOutDegree, stream = myStream)
        new_n_edges = ex_prefix_sum_gpu(newFirstEdge, MAX_TPB = MAX_TPB,
                                        stream = myStream)

        new_n_edges = new_n_edges.getitem(0, stream = myStream)
        new_n_edges = int(new_n_edges)

        # if no edges remain, then MST has converged
        if new_n_edges == 0:
            final_converged = True
            del newOutDegree, newFirstEdge, new_vertex
            break

        # create arrays for new edges
        new_dest = cuda.device_array(new_n_edges, dtype = np.int32,
                                     stream = myStream)
        new_edge_id = cuda.device_array(new_n_edges, dtype = np.int32,
                                        stream = myStream)
        new_weight = cuda.device_array(new_n_edges, dtype = weight.dtype,
                                       stream = myStream)

        top_edge = cuda.device_array_like(newFirstEdge, stream = myStream)
        top_edge.copy_to_device(newFirstEdge, stream = myStream)

        # assign and insert new edges
        assignInsert_CUDA[gridDim, MAX_TPB, myStream](edge_id, dest, weight,
                                            firstEdge, outDegree, colors,
                                            new_vertex, new_dest, new_edge_id,
                                            new_weight, top_edge)

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

    returnLabels = labels.copy_to_host()

    del dest, weight, edge_id, firstEdge, outDegree, converged, labels

    return returnLabels

@cuda.jit
def update_labels_cuda(labels, update_array):
    """
    CUDA version of update_labels.
    """

    v = cuda.grid(1)
    n_components = labels.size

    if v >= n_components:
        return

    curr_color = labels[v]
    labels[v] = update_array[curr_color]

@cuda.jit
def update_labels_single_pass_cuda(labels, colors, new_vertex):
    """
    Does all the updates on a single pass
    """
    v = cuda.grid(1)
    n_components = labels.size

    if v >= n_components:
        return    

    curr_color = labels[v]
    new_color = colors[curr_color]
    new_color_id = new_vertex[new_color]
    labels[v] = new_color_id