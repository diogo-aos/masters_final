# -*- coding: utf-8 -*-
"""
Created on: 01-04-2015
Last modified: 17-09-2015

@author: Diogo Silva

TODO:
- implement cuda distance reduce job
- converge mode in all label functions (low priority since those are not in use)
- improve cuda labels with local block memory
"""

import numpy as np
import numba as nb
from numba import jit, cuda, int32, float32, void

from random import sample

from itertools import islice


from MyML.utils.sorting import arg_k_select


class K_Means:
    '''
    n_clusters              : number of clusters
    tol                     : error tolerance between iterations to stop
    max_iter                : maximum number of iterations
    init                    : 'random' from random sample of dataset, or 
                              a numpy.array with the centroids to use
    label_mode              : 'cuda' for using GPU for cuda,
                              'numba' for using optimized Numba function,
                              'numpy' for using Numpy,
                              'python' for using pure Python function
    centroid_mode           : 'numba' for using optimized Numba function,
                              'numpy' for using Numpy,
                              'python' for using pure Python function
    cuda_mem                : 'manual' for optimized scheme,
                              'auto' for automatic from Numba library

    Several parameters can be configured (aside from _MAX_THREADS_BLOCK and _PPT
    , these value are best left unchanged):
     - _MAX_THREADS_BLOCK : (512) specifies the maximum number of threads per
                            CUDA block
     - _MAX_GRID_XYZ_DIM  : (65535) specifies the maximum dimension of the CUDA
                            block grid in any dimension
     - _CUDA_WARP         : (32) specifis the size of a CUDA warp
     - _PPT               : (1) data points to process per thread, can have
                            significant impact on performance
    '''
       

    def __init__(self, n_clusters=8, tol=1e-4,
                 max_iter=300, init='random', **kwargs):

        self.n_clusters = n_clusters

        self._label_mode = kwargs.get("label_mode", "cuda") #label mode
        self._centroid_mode = kwargs.get("centroid_mode", "numba")#centroid mode

        acceptable_values = {"label":["python", "numpy", "numba", "cuda"],
                             "centroid": ["python", "numpy", "numba"]}
        if self._label_mode not in acceptable_values['label']:
            raise ValueError("label_mode should be one "
                             "of {}".format(acceptable_values['label']))

        if self._centroid_mode not in acceptable_values['centroid']:
            raise ValueError("centroid_mode should be one "
                             "of {}".format(acceptable_values['centroid']))

        # check if centroids are supplied
        if init == 'random':
            self.centroid_init_mode = 'random'
        elif type(init) is np.ndarray:
            if init.shape[0] != n_clusters:
                raise Exception("Number of clusters indicated different \
                                 from number of centroids supplied.")
            self.centroid_init_mode = init.copy()
        else:
            raise ValueError('Centroid  init may be \'random\' or an ndarray \
                              containing the centroids to use.')

        # profiling GPU
        self.gpu_profile = kwargs.get("gpu_profile", False)

        # execution flow
        self.tol = tol
        self.max_iter = max_iter

        # converge with until tol or max_iter
        self._converge = kwargs.get("converge", True)
        self._last_iter = False        

        # outputs
        self.inertia_ = np.inf
        self.iters_ = 0

        # cuda stuff
        self._cudaDataHandle = None
        self._cuda_labels_handle = None
        self._cuda_dists_handle = None
        
        self._cuda = True
        self._cuda_mem = kwargs.get("cuda_mem", "manual")

        self._dist_kernel = 0 # 0 = normal index, 1 = special grid index
        
        self._gridDim = None
        self._blockDim = None
        self._MAX_THREADS_BLOCK = 512
        self._MAX_GRID_XYZ_DIM = 65535
        self._CUDA_WARP = 32

        self._PPT = 1 # points to process per thread

    def fit(self, data):

        if data.dtype != np.float32:
            raise Warning("DATA DUPLICATION: data converted to float32."
                          "TODO: accept other formats")
            data = data.astype(np.float32)

        # GPU profile
        self._set_up_profiling()
        
        N,D = data.shape
            
        self.N = N
        self.D = D
        
        # if random centroids, than get them otherwise they're already there
        if self.centroid_init_mode == 'random':
            self.centroids = self._init_centroids(data)

        elif type(self.centroid_init_mode) is np.ndarray:
            if self.centroid_init_mode.shape[0] != n_clusters:
                raise ValueError("Number of clusters indicated different "
                                  "from number of centroids supplied.")
            self.centroids = self.centroid_init_mode.copy()

        else:
            raise ValueError("Centroid  init may be \'random\' or an ndarray "
                             "containing the centroids to use.")

        # reset variables for flow control
        stopcond = False
        self.iters_ = 0 # iteration count
        self.inertia_ = np.inf # sum of distances
        self._last_iter = False # this is only for labels centroid recomputation

        self._dists = np.empty(N, dtype=np.float32)

        while not stopcond:
            # compute labels
            labels = self._label(data, self.centroids)

            self.iters_ += 1 #increment iteration counter

            ## evaluate stop conditions
            # convergence condition
            if self._converge:
                # compute new inertia
                new_inertia = self._dists.sum()

                # compute error
                error = np.abs(new_inertia - self.inertia_)
                self._error = error
                # save new inertia
                self.inertia_ = new_inertia

                # stop if convergence tolerance achieved
                if error <= self.tol:
                    stopcond = True
                    self._last_iter = True

            # iteration condition
            if self.iters_ >= self.max_iter:
                stopcond = True
                self._last_iter = True

            if stopcond:
                break
            # compute new centroids
            self.centroids = self._recompute_centroids(data, self.centroids,
                                                       labels)
        
        self.labels_ = labels
        self.cluster_centers_ = self.centroids

    def _init_centroids(self, data):
        
        #centroids = np.empty((self.n_clusters,self.D),dtype=data.dtype)
        #random_init = np.random.randint(0,self.N,self.n_clusters)
        
        random_init = sample(xrange(self.N), self.n_clusters)
        #self.init_seed = random_init

        centroids = data[random_init]

        return centroids
 
    def reset_timers(self):
        del self.man_prof
        del self.auto_prof

    def _set_up_profiling(self):
        # set up profiling for manual gpu mem management
        if self._cuda_mem == 'manual':
            if not hasattr(self, 'man_prof'):
                man_prof_events = ('data_ev1', 'data_ev2',
                                   'labels_ev1', 'labels_ev2',
                                   'dists_ev1', 'dists_ev2',
                                   'centroids_ev1', 'centroids_ev2',
                                   'kernel_ev1', 'kernel_ev2')
                self.man_prof = {key:cuda.event() for key in man_prof_events}
                man_prof_timings = ('data_timings', 'labels_timings',
                                    'dists_timings', 'kernel_timings',
                                    'centroids_timings')
                self.man_prof.update({key:list() for key in man_prof_timings})

        # set up profiling for auto gpu mem management
        elif self._cuda_mem == 'auto':
            if not hasattr(self, 'auto_prof'):
                auto_prof_events = ('kernel_ev1', 'kernel_ev2')
                self.auto_prof = {key:cuda.event() for key in auto_prof_events}
                self.auto_prof['kernel_timings']=list()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          LABEL METHODS                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _label(self, data, centroids):
        """
        results is a tuple of labels (pos 0) and distances (pos 1) when
        self._converge == True
        """

        # we need array for distances to check convergence
        if self._label_mode == "cuda":
            labels = self._cu_label(data, centroids)

        elif self._label_mode == "special": #for tests only
            labels=np.empty(self.N, dtype=np.int32)
            self._cu_label_kernel(data,centroids,labels,[1,512],[1,59])

        elif self._label_mode == "numpy":
            labels = np_label(data, centroids, self._dists, self._converge)

        elif self._label_mode == "numba":
            labels = numba_label(data, centroids, self._dists, self._converge)

        elif self._label_mode == "python":
            labels = py_label(data, centroids, self._dists, self._converge)

        return labels

    def _compute_cuda_dims(self, data, use2d = False):

        N, D = data.shape

        if use2d:
            blockHeight = self._MAX_THREADS_BLOCK
            blockWidth = 1
            blockDim = blockWidth, blockHeight

            # threads per block
            tpb = np.prod(blockDim)

            # blocks per grid = data cardinality divided by number
            # of threads per block (1 thread - 1 data point)
            bpg = np.int(np.ceil(np.float(N) / tpb)) 


            # if grid dimension is bigger than MAX_GRID_XYZ_DIM,
            # the grid columns must be broken down in several along
            # the other grid dimensions
            if bpg > self._MAX_GRID_XYZ_DIM:
                # number of grid columns
                gridWidth = np.ceil(bpg / self._MAX_GRID_XYZ_DIM)
                # number of grid rows
                gridHeight = np.ceil(bpg / gridWidth)    

                gridDim = np.int(gridWidth), np.int(gridHeight)
            else:
                gridDim = 1,bpg
        else:
            blockDim = self._MAX_THREADS_BLOCK
            points_in_block = self._MAX_THREADS_BLOCK * self._PPT
            bpg = np.float(N) / points_in_block
            gridDim = np.int(np.ceil(bpg))
            
            
        self._blockDim = blockDim
        self._gridDim = gridDim
    
    def _cu_label(self, data, centroids):
        #WARNING: data is being transposed when sending to GPU

        data_ev1, data_ev2 = cuda.event(), cuda.event()
        labels_ev1, labels_ev2 = cuda.event(), cuda.event()
        dists_ev1, dists_ev2 = cuda.event(), cuda.event()

        N,D = data.shape
        K,cD = centroids.shape
        
        if self._cuda_mem not in ('manual','auto'):
            raise Exception("cuda_mem = \'manual\' or \'auto\'")
            
        if self._gridDim is None or self._blockDim is None:
            self._compute_cuda_dims(data)       
        
        labels = np.empty(N, dtype = np.int32)

        if self._cuda_mem == 'manual':
            # copy dataset and centroids, allocate memory

            ## cuda persistent handles
            # avoids redundant data transfer
            # if dataset has not been sent to device, send it and save handle
            if self._cudaDataHandle is None:
                dataT = np.ascontiguousarray(data.T)

                self.man_prof['data_ev1'].record()
                dData = cuda.to_device(dataT)
                self.man_prof['data_ev2'].record()
                self.man_prof['data_ev2'].synchronize()
                time_ms = cuda.event_elapsed_time(self.man_prof['data_ev1'], 
                                                  self.man_prof['data_ev2'])
                self.man_prof['data_timings'].append(time_ms)
                self._cudaDataHandle = dData
            # otherwise just use handle
            else:
                dData = self._cudaDataHandle

            # avoids creating labels array in device more than once
            if self._cuda_labels_handle is None:
                dLabels = cuda.device_array_like(labels)
                self._cuda_labels_handle = dLabels
            else:
                dLabels = self._cuda_labels_handle

            # avoids creating dists array in device more than once
            if self._cuda_dists_handle is None:
                dDists = cuda.device_array_like(self._dists)
                self._cuda_dists_handle = dDists
            else:
                dDists = self._cuda_dists_handle

            # copy centroids to device
            self.man_prof['centroids_ev1'].record()
            dCentroids = cuda.to_device(centroids)
            self.man_prof['centroids_ev2'].record()
            
            # launch kernel
            self.man_prof['kernel_ev1'].record()
            _cu_label_kernel_dists[self._gridDim,self._blockDim](dData, 
                                                                 dCentroids, 
                                                                 dLabels, 
                                                                 dDists)
            self.man_prof['kernel_ev2'].record()

            # cuda.synchronize()

            # self.man_prof['kernel_ev2'].synchronize()

            # copy labels from device to host
            self.man_prof['labels_ev1'].record()
            dLabels.copy_to_host(ary = labels)
            self.man_prof['labels_ev2'].record()
            
            # copy distance to centroids from device to host
            self.man_prof['dists_ev1'].record()
            dists = dDists.copy_to_host()
            self.man_prof['dists_ev2'].record()
            self._dists = dists

            # synchronize host with gpu before computing times
            self.man_prof['dists_ev2'].synchronize()

            # store timings
            time_ms = cuda.event_elapsed_time(self.man_prof['centroids_ev1'], 
                                              self.man_prof['centroids_ev2'])
            self.man_prof['centroids_timings'].append(time_ms)

            time_ms = cuda.event_elapsed_time(self.man_prof['kernel_ev1'], 
                                              self.man_prof['kernel_ev2'])
            self.man_prof['kernel_timings'].append(time_ms)

            time_ms = cuda.event_elapsed_time(self.man_prof['labels_ev1'], 
                                              self.man_prof['labels_ev2'])
            self.man_prof['labels_timings'].append(time_ms)

            time_ms = cuda.event_elapsed_time(self.man_prof['dists_ev1'], 
                                              self.man_prof['dists_ev2'])
            self.man_prof['dists_timings'].append(time_ms)

        elif self._cuda_mem == 'auto':
            self.auto_prof['kernel_ev1'].record()
            _cu_label_kernel_dists[self._gridDim,self._blockDim](data, 
                                                                centroids, 
                                                                labels, 
                                                                self._dists)
            self.auto_prof['kernel_ev2'].record()
            time_ms = cuda.event_elapsed_time(self.auto_prof['kernel_ev1'], 
                                              self.auto_prof['kernel_ev2'])
            self.auto_prof['kernel_timings'].append(time_ms)

        else:
            raise ValueError("CUDA memory management type may either \
                              be \'manual\' or \'auto\'.")
        
        return labels
        
    def _cu_label_kernel(self, a, b, c, d, gridDim, blockDim):
        """
        Wraper to choose between kernels.
        """
        # if converging and manual memory management, use distance handle
        if self._cuda_mem == 'manual':
            self._cu_label_kernel_dists[gridDim,blockDim](a,b,c,d)
        # if converging and auto memory management, use distance array
        else:
            self._cu_label_kernel_dists[gridDim,blockDim](a,b,c,d)
        pass

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                          CENTROID METHODS                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _recompute_centroids(self, data, centroids, labels):
        if self._centroid_mode == "python":
            centroid_fn = py_recompute_centroids

        elif self._centroid_mode == "numpy":
            centroid_fn = np_recompute_centroids

        elif self._centroid_mode == "numba":
            centroid_fn = numba_recompute_centroids

        else:
            raise ValueError("centroid mode invalid:", self._centroid_mode)

        new_centroids = centroid_fn(data, centroids, labels, self._dists)

        return new_centroids



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   LABEL ALGORITHMS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def py_label(data, centroids, dists, converge):


    N = data.shape[0]
    K,D = centroids.shape

    labels = np.empty(N, dtype=np.int32)

    for n in range(0, N):

        # first iteration outside loop
        dist = 0.0
        for d in range(D): # compute distance
            diff = data[n,d] - centroids[0,d]
            dist += diff ** 2

        best_dist = dist
        best_label = 0

        # remaining iterations
        for k in range(1, K):

            dist = 0.0
            for d in range(D): # compute distance
                diff = data[n,d] - centroids[k,d]
                dist += diff ** 2

            if dist < best_dist:
                best_dist = dist
                best_label = k

        labels[n] = best_label

        if converge:
            dists[n] = best_dist

    return labels

numba_label = nb.njit(py_label)
        
def np_label(data, centroids, dists, converge):
    """ uses more memory because of temporaries
    """


    N,D = data.shape
    C,cD = centroids.shape

    labels = np.zeros(N,dtype=np.int32)

    # first iteration of all datapoints outside loop
    # distance from points to centroid 0
    best_dist = data - centroids[0]
    best_dist = best_dist ** 2
    best_dist = best_dist.sum(axis=1) 


    for c in xrange(1,C):
        # distance from points to centroid c
        dist = data - centroids[c]
        dist = dist ** 2
        dist = dist.sum(axis=1)
        
        #thisCluster = np.full(N,c,dtype=np.int32)
        #labels = np.where(dist < bestd_ist,thisCluster,labels)
        labels[dist < best_dist] = c
        best_dist = np.minimum(dist, best_dist)

    if converge:
       dists = best_dist

    return labels


# data, centroids, labels
@cuda.jit("void(float32[:,:], float32[:,:], int32[:])")
def _cu_label_kernel_normal(a,b,c):
    """
    Computes the labels of each data point without storing the distances.
    """


    # thread ID inside block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # block ID
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # block dimensions
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    # grid dimensions
    gw = cuda.gridDim.x
    gh = cuda.gridDim.y

    # compute thread's x and y index (i.e. datapoint and cluster)
    # tx doesn't matter
    # the second column of blocks means we want to add
    # 2**16 to the index
    n = ty + by * bh + bx*gh*bh

    N = c.shape[0] # number of datapoints
    K,D = b.shape # centroid shape

    if n >= N:
        return

    # first iteration outside loop
    dist = 0.0
    for d in range(D):
        diff = a[n,d]-b[0,d]
        dist += diff ** 2

    best_dist = dist
    best_label = 0

    # remaining iterations
    for k in range(1,K):

        dist = 0.0
        for d in range(D):
            diff = a[n,d]-b[k,d]
            dist += diff ** 2

        if dist < best_dist:
            best_dist = dist
            best_label = k

    c[n] = best_label


CUDA_PPT = 2
# data, centroids, labels
@cuda.jit("void(float32[:,:], float32[:,:], int32[:], float32[:])")
def _cu_label_kernel_dists(a,b,c,dists):

    """
    Computes the labels of each data point storing the distances.
    """

    # # thread ID inside block
    tgid = cuda.grid(1) * CUDA_PPT

    N = c.shape[0] # number of datapoints
    K,D = b.shape # centroid shape

    if tgid >= N:
        return

    for n in range(tgid, tgid + CUDA_PPT):

        if n >= N:
            return

        # first iteration outside loop
        dist = 0.0
        for d in range(D):
            diff = a[d,n] - b[0,d]
            dist += diff ** 2

        best_dist = dist
        best_label = 0

        # remaining iterations
        for k in range(1,K):

            dist = 0.0
            for d in range(D):
                diff = a[d,n]-b[k,d]
                dist += diff ** 2


            if dist < best_dist:
                best_dist = dist
                best_label = k

        c[n] = best_label
        dists[n] = best_dist


# @cuda.jit("void(float32[:,:], float32[:,:], int32[:], float32[:])")
# def _cu_label_kernel_dists_sm(data,centroids,labels,dists):

#     """
#     Computes the labels of each data point storing the distances.
#     Data in c major order.
#     Copy each centroid to shared memory, to a maximum of 4096 dimensions.
#     """
#     # at runtime the number of dimensions must be passed
#     sm_centroid = cuda.shared.array(shape=0, dtype=float32)

#     # # thread ID inside block
#     tid = cuda.threadIdx.x
#     tgid = cuda.grid(1) * CUDA_PPT

#     N = labels.shape[0] # number of datapoints
#     K,D = centroids.shape # centroid shape

#     if tgid >= N:
#         return

#     dims_per_thread = math.ceil(1.0 * K / D)
#     dim_init = tid * dims_per_thread

#     # load first centroid
#     # compute dims to copy

#     for i in range(dim_init, dim_init + dims_per_thread):
#         if i >= K:
#             break
#         sm_centroid[0,i] = centroids[0,i]

#     cuda.syncthreads()

#     for n in range(tgid, tgid + CUDA_PPT):
#         if n >= N:
#             return

#         # first iteration outside loop
#         dist = 0.0
#         for d in range(D):
#             diff = data[d,n] - sm_centroid[0,d]
#             dist += diff ** 2

#         best_dist = dist
#         best_label = 0

#     for k in range(1,K):

#         # load centroid
#         for i in range(dim_init, dim_init + dims_per_thread):
#             if i >= K:
#                 break
#             sm_centroid[0,i] = centroids[0,i]

#         cuda.syncthreads()        

#         # remaining iterations
#         for n in range(tgid, tgid + CUDA_PPT):

#             dist = 0.0
#             for d in range(D):
#                 diff = data[d,n] - centroids[k,d]
#                 dist += diff ** 2

#             if dist < best_dist:
#                 best_dist = dist
#                 best_label = k

#         labels[n] = best_label
#         dists[n] = best_dist


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   CENTROID RECOMPUTATION ALGORITHMS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def py_recompute_centroids(data, centroids, labels, dists):
    N = labels.size
    K, D = centroids.shape

    new_centroids = np.zeros((K,D), dtype=np.float32)       

    # count samples in clusters
    labels_bincount = np.zeros(K, dtype=np.int32)
    for n in xrange(N):
        l = labels[n]
        labels_bincount[l] += 1

    # check for empty clusters
    n_emptyClusters = 0
    for l in xrange(K):
        if labels_bincount[l] == 0:
            n_emptyClusters += 1

    # if there are N empty clusters, we will get the N most distant points
    # from their centroids to attribute to this empty clusters
    if n_emptyClusters > 0:
        # get farthest points from clusters (K-select)
        furtherDistsArgs = np.empty(n_emptyClusters, dtype=np.int32) 
        arg_k_select(dists, n_emptyClusters, furtherDistsArgs)
        # furtherDistsArgs = arg_k_select(dists, n_emptyClusters)

    # increment datapoints to respective centroids
    for n in xrange(N):
        n_label = labels[n]
        for d in xrange(D):
            new_centroids[n_label,d] += data[n,d]

    i = 0
    for k in xrange(K):
        if labels_bincount[k] != 0: # compute final centroid
            for d in xrange(D):
                new_centroids[k, d] /= labels_bincount[k]
        else: # centroid will be one of furthest points
            i_arg = furtherDistsArgs[i]
            for d in xrange(D):
                new_centroids[k, d] = data[i_arg, d]
            i += 1

    return new_centroids

numba_recompute_centroids = nb.njit(py_recompute_centroids)

def np_recompute_centroids(data, centroids, labels, dists):
    """
    this version doesn't discard clusters; instead it uses the same scheme
    as scikit-learn
    """
    # change to get dimension from class or search a non-empty cluster
    #dim = grouped_data[0][0].shape[1]
    N,D = data.shape
    K,D = centroids.shape       
    
    #new_centroids = centroids.copy()
    new_centroids = np.zeros_like(centroids)

    nonEmptyClusters = np.unique(labels)

    n_emptyclusters = K - nonEmptyClusters.size
    furtherDistsArgs = dists.argsort()[::-1][:n_emptyclusters]

    j=0 #empty cluster indexer
    for i in xrange(K):
        if i in nonEmptyClusters:
            new_centroids[i] = data[labels==i].mean(axis=0)
        else:
            new_centroids[i] = data[furtherDistsArgs[j]]
            j+=1

    return new_centroids


if __name__ == '__main__':

    n = 10000
    d = 200
    k = 50

    data = np.random.random((n,d)).astype(np.float32)
    centroids = np.random.random((k,d)).astype(np.float32)

    kt_start, kt_end = cuda.event(), cuda.event()

    # grid config
    tpb = 256
    bpg = np.int(np.ceil(np.float(n) / tpb))

    # compile kernel
    dData = cuda.to_device(data)
    dCentroids = cuda.to_device(centroids)
    dLabels = cuda.device_array(n, dtype=np.int32)
    dDists = cuda.device_array(n, dtype=np.float32)
    _cu_label_kernel_dists[bpg,tpb](dData, dCentroids, dLabels, dDists)

    ## data column major
    # GPU data
    data_t = data.T
    dData = cuda.to_device(data_t)
    dCentroids = cuda.to_device(centroids)
    dLabels = cuda.device_array(n, dtype=np.int32)
    dDists = cuda.device_array(n, dtype=np.float32)

    # kernel
    kt_start.record()
    _cu_label_kernel_dists[bpg,tpb](dData, dCentroids, dLabels, dDists)
    kt_end.record()
    kt_end.synchronize()

    # time
    time_ms = cuda.event_elapsed_time(kt_start, kt_end)
    print 'Kernel time (data column major):{} ms'.format(time_ms)

    ## data row major
    # GPU data
    dData = cuda.to_device(data)
    dCentroids = cuda.to_device(centroids)
    dLabels = cuda.device_array(n, dtype=np.int32)
    dDists = cuda.device_array(n, dtype=np.float32)

    # kernel
    kt_start.record()
    _cu_label_kernel_dists[bpg,tpb](dData, dCentroids, dLabels, dDists)
    kt_end.record()
    kt_end.synchronize()

    # time
    time_ms = cuda.event_elapsed_time(kt_start, kt_end)
    print 'Kernel time (data row major):{} ms'.format(time_ms)


