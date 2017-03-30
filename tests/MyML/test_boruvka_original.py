# -*- coding: utf-8 -*-

"""
author: Diogo Silva
Tests for MST algorithms.
"""

from MyML.utils.profiling import Timer

tm = Timer()

tm.tic()

import tables
import numpy as np
from numba import cuda, njit, jit, int32, float32
from scipy.sparse.csr import csr_matrix
from MyML.graph.mst import boruvka_minho_seq, boruvka_minho_gpu
from MyML.graph.connected_components import connected_comps_seq as getLabels_seq,\
                                         connected_comps_gpu as getLabels_gpu
from MyML.helper.scan import exprefixsumNumba

import sys
import socket
import os.path

tm.tac()
print "Time to load modules (compile some numba stuff): ", tm.elapsed


hostname = socket.gethostname()
mighty4 = "/home/diogoaos/"
mariana = "/home/courses/aac2015/diogoaos/"
dove = "/home/chiroptera/"
if 'dove' in hostname:
    home = dove
elif 'mariana' in hostname:
    home = mariana
elif 'Mighty4' in hostname:
    home = mighty4

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                     UTILS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@njit
def outdegree_from_firstedge(firstedge, outdegree, n_edges):
    n_vertices = firstedge.size
    for v in range(n_vertices - 1):
        outdegree[v] = firstedge[v + 1] - firstedge[v]
    outdegree[n_vertices - 1] = n_edges - firstedge[n_vertices - 1]


def special_bfs(dest, fe, od, mst):

    undiscovered = set(range(fe.size))
    queue = [0]
    n_mst = 1

    while len(undiscovered) != 0:
        vertex = queue.pop()
        
        start, end = fe[vertex], fe[vertex] + od[vertex]
        for edge in range(start, end):
            dest_vertex = dest[edge]
            if dest_vertex not in discovered:
                queue.append(dest_vertex)
                undiscovered.remove(dest_vertex)

        if len(queue) == 0 and len(undiscovered) != 0:
            queue.append(undiscovered.pop())
            n_mst += 1

    return n_mst

@cuda.jit
def newOutDegree(mst, dest, fe):
    v = cuda.grid(1)
    pass

@jit
def binaryEdgeIdSearch(key, dest, fe, od):
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

@jit(["int32(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])",
      "int32(int32[:], float32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], float32[:])"], nopython=True)
def get_new_graph(dest, weight, fe, od, mst, nod, nfe, ndest, nweight):

    # first build the outDegree to get the first_edge
    for e in range(mst.size):
        edge = mst[e]
        o_v = dest[edge] # destination
        i_v = binaryEdgeIdSearch(edge, dest, fe, od)
        if i_v == -1:
            return -1
        nod[o_v] += 1
        nod[i_v] += 1

    # get first edge from outDegree
    exprefixsumNumba(nod, nfe, init = 0)

    #get copy of newFirstEdge to serve as pointers for the newDest
    top_edge = np.empty(nfe.size, dtype = np.int32)
    for i in range(nfe.size):
        top_edge[i] = nfe[i]
    #top_edge = nfe.copy()

    # go through all the mst edges again and write the new edges in the new arrays
    for e in range(mst.size):
        edge = mst[e]

        o_v = dest[edge] # destination vertex
        i_v = binaryEdgeIdSearch(edge, dest, fe, od)
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

def load_csr_graph(filename):
    """
    Loads graph from a file. Every line is of the format "V_origin,V_destination,Edge_weight".
    Returns a scipy.sparse.csr_matrix with the data.
    """
    raw = np.genfromtxt(filename, delimiter = ",", dtype = np.int32)
    sp_raw = csr_matrix((raw[:,2],(raw[:,0],raw[:,1])))
    return sp_raw

def load_h5_to_csr(filename):
    f = tables.open_file(filename, 'r')
    dest = f.root.destination.read()
    origin = f.root.origin.read()
    weight = f.root.weight.read()
    sp_raw = csr_matrix((weight, (origin,dest)))
    f.close()
    return sp_raw

def get_boruvka_format(csr_mat):
    """
    Receives a scipy.sparse.csr_matrix with the graph and outputs
    the 4 components necessary for the full representation of the
    graph for the Boruvka algorithm.
    """
    dest = csr_mat.indices
    weight = csr_mat.data
    firstEdge = csr_mat.indptr[:-1]
    outDegree = np.empty_like(firstEdge)
    outdegree_from_firstedge(firstEdge, outDegree, dest.size)
    return dest, weight, firstEdge, outDegree


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                     GRAPHS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # #

simple_graph = dict()
simple_graph["dest"] = np.array([1, 3, 2, 0, 3, 0, 3, 0, 1, 2, 5, 4, 6, 7, 5, 7, 6, 7], dtype = np.int32)
simple_graph["weight"] = np.array([2, 2, 1, 2, 3, 1, 3, 2, 3, 3, 1, 1, 3, 7, 3, 2, 2, 7], dtype = np.float32)
simple_graph["firstEdge"] = np.array([0, 3, 5, 7, 10, 11, 14, 16], dtype = np.int32)
simple_graph["outDegree"] = np.array([3, 2, 2, 3, 1, 3, 2, 2], dtype = np.int32)

# # # # # # # # # # # # # # # # # #
simple_graph_connect = dict()
simple_graph_connect["dest"] = np.array([1, 2, 3, 2, 0, 2, 0, 1, 7, 5, 4, 6, 7, 5, 7, 3, 5, 6], dtype = np.int32)
simple_graph_connect["weight"] = np.array([3, 1, 2, 1, 2, 3, 2, 3, 3, 1, 1, 3, 7, 3, 2, 3, 7, 2], dtype = np.float32)
simple_graph_connect["firstEdge"] = np.array([0, 3, 4, 6, 9, 10, 13, 15], dtype = np.int32)
simple_graph_connect["outDegree"] = np.array([3, 1, 2, 3, 1, 3, 2, 3], dtype = np.int32)

# # # # # # # # # # # # # # # # # #

four_elt_mat = np.genfromtxt(home + "QCThesis/datasets/graphs/4elt.edges", delimiter=" ",
                              dtype=[("firstedge","i4"),("dest","i4"),("weight","f4")],
                              skip_header=1)
four_elt_mat_s = csr_matrix((four_elt_mat["weight"], (four_elt_mat["firstedge"], four_elt_mat["dest"])))

del four_elt_mat

four_elt = dict()
four_elt["dest"] = four_elt_mat_s.indices
four_elt["weight"] = four_elt_mat_s.data
four_elt["firstEdge"] = four_elt_mat_s.indptr[:-1]
four_elt["outDegree"] = np.empty_like(four_elt["firstEdge"])

del four_elt_mat_s

outdegree_from_firstedge(four_elt["firstEdge"], four_elt["outDegree"], four_elt["dest"].size)

# # # # # # # # # # # # # # # # # # 


def load_graph(name):
    # simple graph of 8 vertices and 9 edges

    graph_names = {"simple_graph" : simple_graph,
                   "simple_graph_connect" : simple_graph_connect,
                   "4elt" : four_elt}

    if name not in graph_names.keys():
        raise Exception("GRAPH " + name + " DOES NOT EXIST.")
    else:
        graph = graph_names[name]

        return graph["dest"], graph["weight"], graph["firstEdge"], graph["outDegree"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                     TESTS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def host_boruvka():

    print "HOST CPU BORUVKA"

    dest, weight, firstEdge, outDegree = load_graph("4elt")

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

def device_boruvka():

    print "CUDA BORUVKA"

    dest, weight, firstEdge, outDegree = load_graph("4elt")

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_gpu(dest, weight, firstEdge, outDegree)
    t1.tac()

    if n_edges < mst.size:
        mst = mst[:n_edges]    

    print "time elapsed: ", t1.elapsed
    mst.sort()
    print mst
    print n_edges

def host_vs_device():
    print "HOST VS DEVICE"

    same_sol = list()
    same_cost = list()

    for r in range(20):
        dest, weight, firstEdge, outDegree = load_graph("4elt")

        t1, t2 = Timer(), Timer()

        t1.tic()
        mst1, n_edges1 = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
        t1.tac()

        if n_edges1 < mst1.size:
            mst1 = mst1[:n_edges1]
        mst1.sort()

        t2.tic()
        mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstEdge, outDegree, MAX_TPB=256)
        t2.tac()

        if n_edges2 < mst2.size:
            mst2 = mst2[:n_edges2]
        mst2.sort()

        same_sol.append(np.in1d(mst1,mst2).sum())
        same_cost.append(weight[mst1].sum() == weight[mst2].sum())
        #same_sol.append((mst1==mst2).all())

    print "no. edges: ", weight.size
    print "no. nodes: ", firstEdge.size

    print "Same solution: ", same_sol
    print "Same cost:", np.all(same_cost)

    print "Solution CPU cost: ", weight[mst1].sum()
    print "Solution GPU cost: ", weight[mst2].sum()

    print "Host time elapsed:   ", t1.elapsed
    print "Device time elapsed: ", t2.elapsed


def check_colors():

    print "CHECK COLORS SEQ & CUDA"

    #dest, weight, firstEdge, outDegree = load_graph("4elt")

    sp_cal = load_csr_graph(home + "QCThesis/datasets/graphs/USA-road-d.CAL.csr")
    dest, weight, firstEdge, outDegree = get_boruvka_format(sp_cal)
    del sp_cal

    print "# edges:            ", dest.size
    print "# vertices:         ", firstEdge.size
    print "size of graph (MB): ", (dest.size + weight.size + firstEdge.size + outDegree.size) * 4.0 / 1024 / 1024    

    print "# vertices: ", firstEdge.size
    print "# edges:    ", dest.size

    print "seq: Computing MST"

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
    t1.tac()

    print "seq: time elapsed: ", t1.elapsed
    print "seq: mst size :", mst.size
    print "seq: n_edges: ", n_edges


    if n_edges < mst.size:
        mst = mst[:n_edges]
    mst.sort()

    print "gpu: Computing MST"

    t1.tic()
    mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstEdge, outDegree, MAX_TPB=256)
    t1.tac()

    print "gpu: time elapsed: ", t1.elapsed
    print "gpu: mst size :", mst2.size  
    print "seq: n_edges: ", n_edges2

    if n_edges2 < mst2.size:
        mst2 = mst2[:n_edges2]
    mst2.sort()


    print "mst gpu == seq: ", (mst == mst2).all()

    # make two cuts
    mst = mst[:-2]

    print "seq: Generating MST graph"
    nod = np.zeros(outDegree.size, dtype = outDegree.dtype)
    nfe = np.empty(firstEdge.size, dtype = firstEdge.dtype)
    ndest = np.empty(mst.size * 2, dtype = dest.dtype)
    nweight = np.empty(mst.size * 2, dtype = weight.dtype)

    t1.tic()
    get_new_graph(dest, weight, firstEdge, outDegree, mst, nod, nfe, ndest, nweight)
    t1.tac()
     
    print "seq: time elapsed: ", t1.elapsed

    print "seq: Computing labels"
    t1.tic()
    colors = getLabels_seq(ndest, nweight, nfe, nod)
    t1.tac()

    print "seq: time elapsed: ", t1.elapsed
    print "seq: # colors:     ", np.unique(colors).size

    print "gpu: Computing labels"
    t1.tic()
    colors2 = getLabels_gpu(ndest, nweight, nfe, nod, MAX_TPB=256)
    t1.tac()

    print "gpu: time elapsed: ", t1.elapsed
    print "gpu: # colors:     ", np.unique(colors2).size

    print "colors gpu == seq: ", (colors == colors2).all()

def mst_cal():
    sp_cal = load_csr_graph(home + "QCThesis/datasets/graphs/USA-road-d.CAL.csr")
    dest, weight, firstEdge, outDegree = get_boruvka_format(sp_cal)
    del sp_cal

    print "# edges:            ", dest.size
    print "# vertices:         ", firstEdge.size
    print "size of graph (MB): ", (dest.size + weight.size + firstEdge.size + outDegree.size) * 4.0 / 1024 / 1024

    times_cpu = list()
    times_gpu = list()
    equal_mst = list()
    equal_cost = list()
    t1, t2 = Timer(), Timer()

    for r in range(10):
        print "cpu round ", r
        t1.tic()
        mst1, n_edges1 = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
        t1.tac()

        print "finished in ", t1.elapsed

        if n_edges1 < mst1.size:
            mst1 = mst1[:n_edges1]

        print "gpu round ", r

        t2.tic()
        mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstEdge, outDegree, MAX_TPB=512)
        t2.tac()

        print "finished in ", t2.elapsed
        print ""

        if n_edges2 < mst2.size:
            mst2 = mst2[:n_edges2]

        equal_mst.append(np.in1d(mst1,mst2).all())
        equal_cost.append(weight[mst1].sum() == weight[mst2].sum())

        if r > 0:
            times_cpu.append(t1.elapsed)
            times_gpu.append(t2.elapsed)

    print equal_mst
    print equal_cost
    print "average time cpu: ", np.mean(times_cpu)
    print "average time gpu: ", np.mean(times_gpu)

def mst_cluster_coassoc():
    t1,t2 = Timer(), Timer()

    #foldername = "/home/courses/aac2015/diogoaos/QCThesis/datasets/gaussmix1e4/"
    foldername = home + "QCThesis/datasets/gaussmix1e4/"

    print "Loading datasets"

    t1.tic()
    # dest = np.genfromtxt(foldername + "prot_dest.csr", dtype = np.int32, delimiter=",")
    # weight = np.genfromtxt(foldername + "prot_weight.csr", dtype = np.float32, delimiter=",")
    # fe = np.genfromtxt(foldername + "prot_fe.csr", dtype = np.int32, delimiter=",")

    dest = np.genfromtxt(foldername + "full_dest.csr", dtype = np.int32, delimiter=",")
    weight = np.genfromtxt(foldername + "full_weight.csr", dtype = np.float32, delimiter=",")
    fe = np.genfromtxt(foldername + "full_fe.csr", dtype = np.int32, delimiter=",")
    t1.tac()

    print "loading elapsed time : ", t1.elapsed

    fe = fe[:-1]
    od = np.empty_like(fe)
    outdegree_from_firstedge(fe, od, dest.size)

    # fix weights to dissimilarity
    weight = 100 - weight

    print "# edges : ", dest.size
    print "# vertices : ", fe.size
    print "edges/vertices ratio : ", dest.size * 1.0 / fe.size

    t1.tic()
    mst, n_edges = boruvka_minho_seq(dest, weight, fe, od)
    t1.tac()

    print "seq: time elapsed : ", t1.elapsed
    print "seq: mst size :", mst.size
    print "seq: n_edges : ", n_edges

    if n_edges < mst.size:
        mst = mst[:n_edges]
    mst.sort()

    ev1,ev2 = cuda.event(), cuda.event()

    ev1.record()
    d_dest = cuda.to_device(dest)
    d_weight = cuda.to_device(weight)
    d_fe = cuda.to_device(fe)
    d_od = cuda.to_device(od)
    ev2.record()

    send_graph_time = cuda.event_elapsed_time(ev1,ev2)

    t2.tic()
    mst2, n_edges2 = boruvka_minho_gpu(d_dest, d_weight, d_fe, d_od, MAX_TPB=512, returnDevAry = True)
    t2.tac()

    ev1.record()
    mst2 = mst2.copy_to_host()
    n_edges2 = n_edges2.getitem(0)
    ev2.record()

    recv_mst_time = cuda.event_elapsed_time(ev1,ev2)
    print "gpu: send graph time : ", send_graph_time
    print "gpu: time elapsed : ", t2.elapsed    
    print "gpu: rcv mst time : ", recv_mst_time
    print "gpu: mst size :", mst2.size  
    print "seq: n_edges : ", n_edges2

    if n_edges2 < mst2.size:
        mst2 = mst2[:n_edges2]
    mst2.sort()

    if n_edges == n_edges2:
        mst_is_equal = (mst == mst2).all()
    else:
        mst_is_equal = False
    print "mst gpu == seq : ", mst_is_equal

def analyze_graph_from_h5(filename, verbose=False):

    def v_print(vstr):
        if verbose:
            print vstr

    csr_mat = load_h5_to_csr(filename)
    dest, weight, firstEdge, outDegree = get_boruvka_format(csr_mat)
    del csr_mat



    n_e = dest.size
    n_v = firstEdge.size
    mem = (dest.size*dest.itemsize + weight.size*weight.itemsize + firstEdge.size*firstEdge.itemsize + outDegree.size*outDegree.itemsize)/ (1024.0**2)
    print "# edges:            ", n_e
    print "# vertices:         ", n_v
    print "size of graph (MB): ", mem

    times_cpu = list()
    times_gpu = list()
    equal_mst = list()
    equal_cost = list()
    mst_costs = {'cpu':list(), 'gpu':list()}
    t1, t2 = Timer(), Timer()

    for r in range(10):
        v_print('------ Round {} -------'.format(r))
        t1.reset()
        t1.tic()
        mst1, n_edges1 = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
        t1.tac()
        v_print('CPU finished in {} s'.format(t1.elapsed))

        if n_edges1 < mst1.size:
            mst1 = mst1[:n_edges1]

        t2.reset()
        t2.tic()
        mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstEdge, outDegree, MAX_TPB=512)
        t2.tac()
        v_print('GPU finished in {} s'.format(t2.elapsed))


        if n_edges2 < mst2.size:
            mst2 = mst2[:n_edges2]


        mst_costs['cpu'].append(weight[mst1].sum())
        mst_costs['gpu'].append(weight[mst2].sum())
        equal_mst.append(np.in1d(mst1,mst2).all())
        equal_cost.append(weight[mst1].sum() == weight[mst2].sum())

        if r > 0:
            times_cpu.append(t1.elapsed)
            times_gpu.append(t2.elapsed)



    max_cost = max((max(mst_costs['cpu']), max(mst_costs['gpu'])))
    cost_error = map(lambda x: abs(x[0]-x[1]), zip(*mst_costs.values()))
    cost_error = map(lambda x: x/max_cost, cost_error)
    error_threshold = 1e-5

    cpu_str = ''
    for t in times_cpu:
        cpu_str += str(t) + ','

    gpu_str = ''
    for t in times_gpu:
        gpu_str += str(t) + ','

    cpu_costs = ''
    for c in mst_costs['cpu']:
        cpu_costs += str(c) + ','
    gpu_costs = ''
    for c in mst_costs['gpu']:
        gpu_costs += str(c) + ','    

    print 'dataset: {}'.format(os.path.basename(filename))
    print 'CPU times,{},{},{},{}'.format(n_e,n_v,mem,cpu_str[:-1])
    print 'GPU times,{},{},{},{}'.format(n_e,n_v,mem,gpu_str[:-1])
    print 'CPU costs,{},{},{},{}'.format(n_e,n_v,mem,cpu_costs[:-1])
    print 'GPU costs,{},{},{},{}'.format(n_e,n_v,mem,gpu_costs[:-1])    
    print ''
    print 'All equal MSTs: {}'.format(np.all(np.array(equal_mst) == equal_mst[0]))
    print 'All equal costs: {}'.format(np.all(equal_cost))
    print 'All cost errors <= {}: {}'.format(error_threshold, np.all(map(lambda x:x<error_threshold, cost_error)))
    print 'Max normalized error: {}'.format(max(cost_error))

    speedup = np.array(times_cpu) / np.array(times_gpu)

    print 'Times(s)\tMean\tStd\tMax\tMin'
    print 'CPU     \t{:.5F}\t{:.5F}\t{:.5F}\t{:.5F}'.format(np.mean(times_cpu), np.std(times_cpu), np.max(times_cpu), np.min(times_cpu))
    print 'GPU     \t{:.5F}\t{:.5F}\t{:.5F}\t{:.5F}'.format(np.mean(times_gpu), np.std(times_gpu), np.max(times_gpu), np.min(times_gpu))
    print 'SpeedUp \t{:.5F}\t{:.5F}\t{:.5F}\t{:.5F}'.format(np.mean(speedup), np.std(speedup), np.max(speedup), np.min(speedup))
    print 'Error   \t{:.5F}\t{:.5F}\t{:.5F}\t{:.5F}'.format(np.mean(cost_error), np.std(cost_error), np.max(cost_error), np.min(cost_error))


def main(argv):
    valid_args = [0, 1, 2, 3, 4, 5, 6]
    valid_args = map(str,valid_args)
    if len(argv) <= 1 or argv[1] not in valid_args:
        print "0 : test host boruvka"
        print "1 : test device boruvka"
        print "2 : device vs host boruvka"
        print "3 : test device boruvka"
        print "4 : test device boruvka"
    elif argv[1] == "0":
        host_boruvka()
    elif argv[1] == "1":
        device_boruvka()
    elif argv[1] == "2":
        host_vs_device()        
    elif argv[1] == "3":
        check_colors()           
    elif argv[1] == "4":
        mst_cal()         
    elif argv[1] == "5":
        mst_cluster_coassoc()
    elif argv[1] == '6':
        if type(argv[2]) is not str:
            raise TypeError("input file path must be string")
        if '-v' in argv:
            verbose = True
        else:
            verbose = False
        analyze_graph_from_h5(argv[2], verbose)

if __name__ == "__main__":
    main(sys.argv)