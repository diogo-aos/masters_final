# -*- coding: utf-8 -*-
"""
Created on 10-04-2015

@author: Diogo Silva

Evidence accumulation clustering. This module aims to include all
features of the Matlab toolbox plus addressing NxK co-association
matrices.

TODO:
- link everything
- add sanity checks on number of samples of partitions
- robust exception handling
"""

import numpy as np
from random import sample
from numba import jit, njit
# from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.sparse.csgraph import connected_components

import scipy_numba.sparse.csgraph._min_spanning_tree as nbMST
from scipy_numba.spatial.distance import squareform

from MyML.cluster.linkage import scipy_numba_slink_wraper as slink
from MyML.cluster.linkage import labels_from_Z

import sparse as eac_sp
biggest_cluster_size = eac_sp._compute_max_assocs_from_ensemble
EAC_CSR = eac_sp.EAC_CSR

from full import EAC_FULL

import os.path

class EAC():

    def __init__(self, n_samples, **kwargs):
        """
        condensed      : (True) stores only half the co-associations (no
                         redundancy), which means pdist format for full matrix
                         ((n*(n-1))/2 length array).
        assoc_dtype    : (numpy.uint8) datatype of the associations

        sparse         : (False) stores co-associations in a sparse matrix
        sparse_sort_mode
        sparse_max_assocs
        sparse_max_assocs_factor
        sparse_max_assocs_mode
        sparse_keep_degree

        coassoc_store
        coassoc_store_path
        sl_disk
        sl_disk_dir
        """

        self.n_samples = n_samples

        # check if all arguments were passed as a dictionary
        args = kwargs.get("args")
        if args is not None and type(args) == dict:
            kwargs == args

        ## generate ensemble parameters
        self.n_partitions = kwargs.get("n_partitions", 100)
        self.iters = kwargs.get("iters", 3)
        self.n_clusters = kwargs.get("n_clusters", "sqrt")
        self.toFiles = False
        self.toFiles_folder = None

        ## build matrix parameters
        self.condensed = kwargs.get("condensed", True)
        self.kNN = kwargs.get("kNN", False)
        self.assoc_dtype = kwargs.get("assoc_dtype", np.uint8)

        # sparse matrix parameters
        self.sparse = kwargs.get("sparse", False)
        self.sp_sort = kwargs.get("sparse_sort_mode", "surgical")
        self.sp_max_assocs = kwargs.get("sparse_max_assocs", None)
        self.sp_max_assocs_factor = kwargs.get("sparse_max_assocs_factor", 3)
        self.sp_max_assocs_mode = kwargs.get("sparse_max_assocs_mode", "linear")
        self.sp_keep_degree = kwargs.get("sparse_keep_degree", False)

        # if not sparse and not kNN then it is full matrix
        if not self.sparse  and not self.kNN:
            self.full = True
        else:
            self.full = False

        self.coassoc_store = kwargs.get("coassoc_store", False)
        self.coassoc_store_path = kwargs.get("coassoc_store_path", None)

        ## final clustering parameters
        self.linkage = kwargs.get("linkage", "SL")
        self.disk_mst = kwargs.get("sl_disk", False)
        self.disk_dir = kwargs.get("sl_disk_dir", './')

        if self.disk_mst:
            self.coassoc_store = True

        if self.coassoc_store_path is None:
            self.coassoc_store_path = os.path.join(self.disk_dir, "coassoc.h5")

    def _validate_params(self):
        pass

    def generateEnsemble(self):
        pass

    def buildMatrix(self, ensemble):

        if self.sparse:
            if self.sp_max_assocs is None:
                self.sp_max_assocs = biggest_cluster_size(ensemble)
                self.sp_max_assocs *= self.sp_max_assocs_factor
            
            coassoc = EAC_CSR(self.n_samples, max_assocs=self.sp_max_assocs,
                              condensed=self.condensed,
                              max_assocs_type=self.sp_max_assocs_mode,
                              sort_mode=self.sp_sort,
                              dtype=self.assoc_dtype)

            coassoc.update_ensemble(ensemble)
            coassoc._condense(keep_degree = self.sp_keep_degree)
            coassoc.make_diassoc()

            if self.coassoc_store:
                coassoc.store(self.coassoc_store_path, delete=True,
                              indptr_expanded=True,
                              store_degree=self.sp_keep_degree)

            self.store_coassoc = coassoc.store

        elif self.full:
            coassoc = EAC_FULL(self.n_samples, condensed=self.condensed,
                               dtype=self.assoc_dtype)
            coassoc.update_ensemble(ensemble)
            coassoc.get_degree() # get association degree and nnz
            coassoc.make_diassoc()

        elif self.kNN:
            raise NotImplementedError("kNN matrix building has not been "
                                      "included in this version.")

        else:
            raise ValueError("Build matrix: No sparse, no full, no kNN. "
                             "No combination possible.")

        self.coassoc = coassoc

    def finalClustering(self, n_clusters=0):
        if self.sparse:
            if not self.disk_mst:
                n_fclusts, labels = sp_sl_lifetime(self.coassoc.csr,
                                                   self.n_partitions,
                                                   n_clusters=n_clusters)
            else:
                n_fclusts, labels = sp_sl_lifetime_disk(self.coassoc_store_path,
                                                        self.n_partitions,
                                                        n_clusters=0,
                                                        index_dir=self.disk_dir)
        elif self.full:
            n_fclusts, labels = full_sl_lifetime(self.coassoc.coassoc,
                                                 self.n_samples,
                                                 n_clusters=n_clusters)
        elif self.kNN:
            raise NotImplementedError("kNN not included in this version yet.")
        else:
            raise ValueError("Final clustering: No sparse, no full, no kNN."
                             " No combination possible.")

        self.n_fclusts = n_fclusts
        self.labels = labels
        return labels


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # #                      FINAL CLUSTERING                       # # # #
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def sp_sl_lifetime(mat, max_dist=None, n_clusters=0):
    """
    Converts graph weights to dissimilarities if input graph is in 
    similarities. Computes MST (Kruskal) of dissimilarity graph.
    Compute number of disconnected clusters (components).
    Sort MST in increasing order to get equivalent of SL clustering.
    Compute lifetimes if number of clusters is not provided.
    Make necessary cuts to have the desired number of clusters.
    Compute connected components (clusters) after the cuts.

    Inputs:
        graph           : dissimilarity matrix in CSR form.
        max_dist        : maximum valid distance in the matrix. If None
                          (default) it assumes the maximum value present in the
                          input graph.
        n_clusters      : number of clusters to compute. If 0 (default), 
                          use lifetime criteria.
    Outputs:
        n_fclusts       : final number of clusters        
        labels          : final clustering labels
    """
    if max_dist is None:
        max_dist = mat.data.max() + 1
    else:
        max_dist += 1

    # get minimum spanning tree
    mst = nbMST.minimum_spanning_tree(mat)

    # compute number of disconnected components
    n_disconnect_clusters = mst.shape[0] - mst.nnz

    # sort associations by weights
    asort = mst.data.argsort()
    sorted_weights = mst.data[asort]

    if n_clusters == 0:
        cont, max_lifetime = lifetime_n_clusters(sorted_weights)

        if n_disconnect_clusters > 1:
            # add 1 to max_dist as the maximum weight because I also added
            # 1 when converting to diassoc to avoid having zero weights
            disconnect_lifetime = max_dist + 1 - sorted_weights[-1]

            # add disconnected clusters to number of clusters if disconnected
            # lifetime is smaller
            if max_lifetime > disconnect_lifetime:
                cont += n_disconnect_clusters - 1
            else:
                cont = n_disconnect_clusters

        nc_stable = cont
    else:
        nc_stable = n_clusters

    # cut associations if necessary
    if nc_stable > n_disconnect_clusters:
        n_cuts = nc_stable - n_disconnect_clusters
        
        mst.data[asort[-n_cuts:]] = 0
        mst.eliminate_zeros()   

    if nc_stable > 1:
        n_comps, labels = connected_components(mst)
    else:
        labels = np.empty(0, dtype=np.int32)
        n_comps = 1  

    return n_comps, labels

#TODO: compute max_dist from disk coassoc
def sp_sl_lifetime_disk(coassoc_path, max_dist,
                        n_clusters=0, index_dir='/tmp/'):
    """
    Converts graph weights to dissimilarities if input graph is in 
    similarities. Computes MST (Kruskal) of dissimilarity graph.
    Compute number of disconnected clusters (components).
    Sort MST in increasing order to get equivalent of SL clustering.
    Compute lifetimes if number of clusters is not provided.
    Make necessary cuts to have the desired number of clusters.
    Compute connected components (clusters) after the cuts.

    Inputs:
        graph           : dis/similarity matrix in CS form.
        max_val         : maximum value from which dissimilarity will be
                          computed. If False (default) assumes input graph
                          already encodes dissimilarities.
        n_clusters      : number of clusters to compute. If 0 (default), 
                          use lifetime criteria.
    Outputs:
        n_fclusts       : final number of clusters        
        labels          : final clustering labels
    """
    # get minimum spanning tree
    mst = nbMST.minimum_spanning_tree_disk_file(coassoc_path,
                                                index_dir=index_dir)

    # compute number of disconnected components
    n_disconnect_clusters = mst.shape[0] - mst.nnz

    # sort associations by weights
    asort = mst.data.argsort()
    sorted_weights = mst.data[asort]

    if n_clusters == 0:
        cont, max_lifetime = lifetime_n_clusters(sorted_weights)

        if n_disconnect_clusters > 1:
            # add 1 to max_dist as the maximum weight because I also added
            # 1 when converting to diassoc to avoid having zero weights
            disconnect_lifetime = max_dist + 1 - sorted_weights[-1]

            # add disconnected clusters to number of clusters if disconnected
            # lifetime is smaller
            if max_lifetime > disconnect_lifetime:
                cont += n_disconnect_clusters - 1
            else:
                cont = n_disconnect_clusters

        nc_stable = cont
    else:
        nc_stable = n_clusters

    # cut associations if necessary
    if nc_stable > n_disconnect_clusters:
        n_cuts = nc_stable - n_disconnect_clusters
        
        mst.data[asort[-n_cuts:]] = 0
        mst.eliminate_zeros()   

    if nc_stable > 1:
        n_comps, labels = connected_components(mst)
    else:
        labels = np.empty(0, dtype=np.int32)
        n_comps = 1  

    return n_comps, labels

def full_sl_lifetime(mat, n_samples, n_clusters=0):

    # convert to diassoc
    if mat.ndim == 2:
        mat = squareform(mat)

    #Z = linkage(mat, method="single")
    Z = slink(mat, n_samples)

    if n_clusters == 0:

        cont, max_lifetime = lifetime_n_clusters(Z[:,2])

        nc_stable = cont
    else:
        nc_stable = n_clusters

    if nc_stable > 1:
        labels = labels_from_Z(Z, n_clusters=nc_stable)
        # rename labels
        i=0
        for l in np.unique(labels):
            labels[labels == l] = i
            i += 1        
    else:
        labels = np.empty(0, dtype=np.int32)

    return nc_stable, labels

def lifetime_n_clusters(weights):
    # compute lifetimes
    lifetimes = weights[1:] - weights[:-1]

    # maximum lifetime
    m_index = np.argmax(lifetimes)
    th = lifetimes[m_index]

    # get number of clusters from lifetimes
    indices = np.where(weights >weights[m_index])[0]
    if indices.size == 0:
        cont = 1
    else:
        cont = indices.size + 1

    #testing the situation when only 1 cluster is present
    # if maximum lifetime is smaller than 2 times the minimum
    # don't make any cuts (= 1 cluster)
    # max>2*min_interval -> nc=1
    close_to_zero_indices = np.where(np.isclose(lifetimes, 0))
    minimum = np.min(lifetimes[close_to_zero_indices])

    if th < 2 * minimum:
        cont = 1

    return cont, th