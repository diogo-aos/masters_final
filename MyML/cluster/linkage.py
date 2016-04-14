# -*- coding: utf-8 -*-
"""
Created on 15-06-2015

@author: Diogo Silva

"""
import numpy as np
from numba import cuda, jit, void, int32, float32

import scipy_numba.cluster._hierarchy_eac as hie_eac


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    Functions to perform SL-Linkage on kNN. Functions take a weight matrix
#    and a neighbors matrix. They also take the output matrix in the input.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def knn_slhac(weights, neighbors, Z):
    n_samples, n_neighbors = weights.shape
    
    track = np.arange(n_samples, dtype = np.int32)

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        a_min = weights.argmin()
        pattern, neigh_idx = a_min // n_neighbors, a_min % n_neighbors
        
        # get neighbor
        neigh = neighbors[pattern, neigh_idx]

        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # unconnected clusters
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == np.inf:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    # update the clusters of the samples in track
                    track[track == clust1] = n_samples + Z_pointer
                    track[track == clust2] = n_samples + Z_pointer

                    # add cluster to Z
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = np.inf
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            # update the clusters of the samples in track
            track[track == pattern_track] = n_samples + Z_pointer
            track[track == neigh_track] = n_samples + Z_pointer

            # add cluster to Z
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = np.inf

@jit(nopython=True)
def knn_slhac_fast(weights, neighbors, Z):
    n_samples, n_neighbors = weights.shape
    
    # allocate and fill track array
    # the track array has the current cluster of each pattern
    track = np.empty(n_samples, dtype = np.int32)
    for i in range(n_samples):
        track[i] = i

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        # get the index of the minimum value in the weights matrix
        a_min = weights.argmin()
        pattern = a_min // n_neighbors
        neigh_idx = a_min % n_neighbors

        # get neighbor corresponding to the neighbor index
        neigh = neighbors[pattern, neigh_idx]

        # get clusters of origin and destination
        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # weight = inf means there are no connected patterns
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == np.inf:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    new_clust = n_samples + Z_pointer
                    # update the clusters of the samples in track
                    for i in range(n_samples):
                        i_clust = track[i]
                        if i_clust == pattern_track or i_clust == neigh_track:
                            track[i] = new_clust

                    # add cluster to solution
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = np.inf
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            new_clust = n_samples + Z_pointer
            # update the clusters of the samples in track
            for i in range(n_samples):
                i_clust = track[i]
                if i_clust == pattern_track or i_clust == neigh_track:
                    track[i] = new_clust

            # add cluster to solution
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = np.inf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    Generic SL-Linkage
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def scipy_numba_slink_wraper(weights, n):
    hie_eac.dists_dtype = weights.dtype
    Z = np.empty((n-1,4), dtype=np.float32)
    hie_eac.slink(weights, Z, n)
    return Z


def slhac(weights, Z):
    """
    modified from knn_slhac.
    """
    n_samples, n_neighbors = weights.shape
    max_val = np.iinfo(weights.dtype).max
    
    track = np.arange(n_samples, dtype = np.int32)

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        a_min = weights.argmin()
        pattern, neigh_idx = a_min // n_neighbors, a_min % n_neighbors
        
        # get neighbor
        #neigh = neighbors[pattern, neigh_idx]
        neigh = neigh_idx # redundant naming to keep code

        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # unconnected clusters
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == max_val:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    # update the clusters of the samples in track
                    track[track == clust1] = n_samples + Z_pointer
                    track[track == clust2] = n_samples + Z_pointer

                    # add cluster to Z
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = max_val
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            # update the clusters of the samples in track
            track[track == pattern_track] = n_samples + Z_pointer
            track[track == neigh_track] = n_samples + Z_pointer

            # add cluster to Z
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = max_val

@jit(nopython=True)
def slhac_fast(weights, Z, max_val):
    n_samples, n_neighbors = weights.shape
    
    # allocate and fill track array
    # the track array has the current cluster of each pattern
    track = np.empty(n_samples, dtype = np.int32)
    for i in range(n_samples):
        track[i] = i

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        # get the index of the minimum value in the weights matrix
        a_min = weights.argmin()
        pattern = a_min // n_neighbors
        neigh_idx = a_min % n_neighbors

        # get neighbor corresponding to the neighbor index
        #neigh = neighbors[pattern, neigh_idx]
        neigh = neigh_idx

        # get clusters of origin and destination
        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # weight = max_val means there are no connected patterns
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == max_val:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    new_clust = n_samples + Z_pointer
                    # update the clusters of the samples in track
                    for i in range(n_samples):
                        i_clust = track[i]
                        if i_clust == pattern_track or i_clust == neigh_track:
                            track[i] = new_clust

                    # add cluster to solution
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = max_val
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            new_clust = n_samples + Z_pointer
            # update the clusters of the samples in track
            for i in range(n_samples):
                i_clust = track[i]
                if i_clust == pattern_track or i_clust == neigh_track:
                    track[i] = new_clust

            # add cluster to solution
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = max_val

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    Get final clustering from linkage matrix
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def labels_from_Z(Z, n_clusters):
    n_samples = Z.shape[0] + 1
    
    track = np.arange(n_samples, dtype = np.int32)
    
    Z_pointer = 0
    while Z_pointer != n_samples - n_clusters:
        clust0 = Z[Z_pointer, 0]
        clust1 = Z[Z_pointer, 1]
        
        # update the clusters of the samples in track
        track[track == clust0] = n_samples + Z_pointer
        track[track == clust1] = n_samples + Z_pointer

        Z_pointer += 1

    # rename labels
    i=0
    for l in np.unique(track):
        track[track == l] = i
        i += 1
        
    return track

@jit(nopython=True)
def labels_from_Z_numba(Z, track, n_clusters):
    # track is an array of size n_samples
    n_samples = track.size

    for i in range(n_samples):
        track[i] = i

    Z_pointer = 0
    while Z_pointer != n_samples - n_clusters:
        clust0 = Z[Z_pointer, 0]
        clust1 = Z[Z_pointer, 1]
        

        # update the clusters of the samples in track
        new_clust = n_samples + Z_pointer
        for i in range(n_samples):
            curr_track = track[i]
            if curr_track == clust0 or curr_track == clust1:
                track[i] = new_clust

        Z_pointer += 1

    map_key = np.empty(n_clusters, np.int32)
    map_val = np.empty(n_clusters, np.int32)

    for i in range(n_clusters):
        map_key[i] = -1
        map_val[i] = i

    for l in range(n_samples):
        clust = track[l]

        # search for clust in map
        key = -1
        found = 0
        for k in range(n_clusters):
            if map_key[k] == clust:
                found = 1
                key = k
                break
            elif map_key[k] == -1:
                key = k
                break

        # if not found, add clust to map
        if found == 0:
            map_key[key] = clust

        val = map_val[key]

        track[l] = val

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    Utils
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@jit(nopython=True)
def binary_search(key, ary):
    """
    Inputs:
        key         : value to find
        ary         : sorted arry in which to find the key

    """
    imin = 0
    imax = ary.size

    while imin < imax:
        imid = (imax + imin) / 2
        imid_val = ary[imid]

        # key is before
        if key < imid_val:
            imax = imid
        # key is after
        elif key > imid_val:
            imin = imid + 1
        # key is between first edge of imid and next first edge
        else:
            return imid
    return -1