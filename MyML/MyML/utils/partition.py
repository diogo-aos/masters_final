# -*- coding: utf-8 -*-
"""
Created on 07-05-2015
Modified on 17-09-2015

@author: Diogo Silva

TODO:

"""

import numpy as np
import matplotlib.pyplot as plt
import os.path
import tables

def convertIndexToBin(clusts):
    """Converts partition in list of arrays (one array per cluster) format to
    binary matrix."""

    # clusts is a list of numpy.arrays where each element in
    # in the array is the index of a sample that belongs to that cluster

    if clusts is None:
        raise Exception("A clustering partition must be provided.")

    N = sum([c.size for c in clusts])
    n_clusts = len(clusts)

    clust_out = np.zeros((n_clusts, N), dtype=np.uint8)

    for i, clust in enumerate(clusts):
        clust_out[i,clust] = 1

    return clust_out

def convertClusterStringToBin(clusts):
    """Converts partition in array format to binary matrix.

    Converts a N length array, where the i-th element contains the id of the
    cluster that the i-th sample belongs too, to a CxN binary matrix where each
    row corresponds to a cluster and the j-th column of the i-th row is 1 iff
    the j-th sample belongs to the i-th cluster.

    In the case that cluster ID can be zero then there is an offset of -1 in the
    rows, e.g. the C-th row actually corresponds to the first cluster.

    It is assumed that cluster numbers go from [0, biggest_clust[ or
    [1, biggest_clust[ uninterrupted, otherwise this function does not work.

    clusts         : N length array with the cluster labels of the N samples
    """
    if clusts is None:
        raise Exception("A clustering partition must be provided.")

    u_clusts = np.unique(clusts)
    if u_clusts.size not in (clusts.max(), clusts.max()+1):
        raise ValueError('cluster numbers go from [0, biggest_clust[ or'
                         '[1, biggest_clust[ uninterrupted')
    n_clusts = u_clusts.size
    N = clusts.size

    clust_out = np.zeros((n_clusts, N), dtype=np.uint8)
    offset = -1 if clusts.min() == 1 else 0

    for sample_ind, clust_ind in enumerate(clusts):
        # cluster_ind is never 0 so we need to subtract 1 to index the array
        clust_out[clust_ind+offset, sample_ind] = 1

    return clust_out


def convertClusterStringToIndex(partition):
    """Converts a partition in the string format (array where the i-th value
    is the cluster label of the i-th pattern) to index format (list of arrays,
    there the k-th array contains the pattern indices that belong to the k-th
    cluster)"""
    clusters = np.unique(partition)
    nclusters = clusters.size
    # nclusters = partition.max() # for cluster id = 0, 1, 2, 3, ....

    finalPartition = [None] * nclusters
    for c,l in enumerate(clusters):
        finalPartition[c] = np.where(partition==l)[0].astype(np.uint64)

    return finalPartition

def generateEnsemble(data, generator, n_clusters=20, npartitions=30, iters=3):
    """Generates an ensemble for the data using the generator algorithm
    provided.
    data        : data to be fed to generator algorithm;
    generator   : generator object with a fit method;
    n_clusters  : number of clusters that the generator should use; can be a
                  list with a range of numbers [min, max];
    npartitions : number of partitions that should be generated;
    iters       : number of iterations the generator should run for
    TODO: check if generator has fit method and n_clusters,labels_ attributes

    """
    ensemble = [None] * npartitions

    if type(n_clusters) is list:
        if n_clusters[0] == n_clusters[1]:
            clusterRange = False
            generator.n_clusters = n_clusters[0]
        else:
            clusterRange = True
            min_ncluster = n_clusters[0]
            max_ncluster = n_clusters[1]
    else:
        clusterRange = False
        generator.n_clusters = n_clusters

    generator.max_iter = iters

    for x in xrange(npartitions):
        if clusterRange:
            k = np.random.randint(min_ncluster, max_ncluster)
            generator.n_clusters = k

        generator.fit(data)
        ensemble[x] = convertClusterStringToIndex(generator.labels_)

    return ensemble


def generateEnsembleToFiles(foldername, data, generator, n_clusters=20,
                            npartitions=30, iters=3, fileprefix="",
                            format_str='%d'):
    """
    TODO: check if generator has fit method and n_clusters,labels_ attributes
    """

    if type(n_clusters) is list:
        if n_clusters[0] == n_clusters[1]:
            clusterRange = False
            generator.n_clusters = n_clusters[0]
        else:
            clusterRange = True
            min_ncluster = n_clusters[0]
            max_ncluster = n_clusters[1]
    else:
        clusterRange = False
        generator.n_clusters = n_clusters

    generator.max_iter = iters

    for x in xrange(npartitions):
        if clusterRange:
            k = np.random.randint(min_ncluster,max_ncluster)
            generator.n_clusters = k

        generator.fit(data)
        partition = convertClusterStringToIndex(generator.labels_)
        savePartitionToFile(foldername + fileprefix + "part{}.csv".format(x),
                            partition, format_str)


def savePartitionToFileCSV(filename, partition, format_str='%d'):
    """
    Assumes partition as list of arrays.
    """
    n_clusters = len(partition)
    with open(filename, "w") as pfile:
        for c in xrange(n_clusters):
            cluster_str = ','.join([format_str % sample for sample in partition[c]])
            pfile.writelines(cluster_str + '\n')

def loadEnsembleFromFile(filename):
    abspath = os.path.abspath(filename)

    # check for directory existence
    if not os.path.exists(abspath):
        raise IOError("Path not found:" + os.path.dirname(abspath))

    if not os.path.isfile(abspath):
        raise IOError("Path is not file:" + abspath)

    with open(filename, 'r') as f:
        ensemble_npz = np.load(f)
        ensemble = ensemble_npz['arr_0']
        return ensemble

def loadEnsembleFromFiles(filelist = None, foldername = None):
    if filelist is None and foldername is None:
        raise Exception("EITHER FILELIST OR FOLDERNAME MUST BE SUPPLIED")
    if filelist is None:
        filelist = [os.path.join(root, name)
                           for root, dirs, files in os.walk(foldername)
                           for name in files
                           if "part" in name]
    ensemble = list()
    for filename in filelist:
        ensemble.append(loadPartitionFromFile(filename))
    return ensemble

def loadPartitionFromFile(filename):
    partition = list()
    with open(filename, "r") as pfile:
        for pline in pfile:
            if pline != '':
                partition.append(np.fromstring(pline, dtype=np.int32, sep=','))

    return partition



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                        HDF FILES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def generateEnsembleToFileHDF(file, data, generator, n_clusters=20,
                              npartitions=30, iters=3, progress=False):
    """
    TODO: check if generator(the algorithm to be used) has fit method and
    n_clusters,labels_ attributes
    """
    # open/create file if file is string of path
    if isinstance(file, str):

        abspath = os.path.abspath(file)

        # check for directory existence
        if not os.path.exists(os.path.dirname(abspath)):
            raise IOError("Directory not found:" + os.path.dirname(abspath))

        # create/open file in write mode
        f = tables.openFile(abspath, 'w')
    elif isinstance(file, tables.file.File):
        f = file
    else:
        raise TypeError("file must be string of path or a tables.file.File")

    if type(n_clusters) is list:
        if n_clusters[0] == n_clusters[1]:
            clusterRange = False
            generator.n_clusters = n_clusters[0]
        else:
            clusterRange = True
            min_ncluster = n_clusters[0]
            max_ncluster = n_clusters[1]
    else:
        clusterRange = False
        generator.n_clusters = n_clusters

    generator.max_iter = iters
    bgs = 0
    bgp = -1
    for x in xrange(npartitions):
        if progress:
            yield x

        if clusterRange:
            k = np.random.randint(min_ncluster, max_ncluster)
            generator.n_clusters = k

        generator.fit(data)

        # convert partition to list of arrays
        partition = convertClusterStringToIndex(generator.labels_)

        # update biggest cluster size
        new_bgs = max(map(np.size, partition))
        if new_bgs > bgs:
            bgs = new_bgs
            bgp = x
        # bgs = new_bgs if new_bgs > bgs else bgs
        # bgp = x if new_bgs

        # save partition to file
        savePartitionToFileHDF(f, partition, x)

    # save metadata
    f.root._v_attrs.N = data.shape[0] # number of samples
    f.root._v_attrs.dim = data.shape[1] # dimension
    # save kmin and kmax
    if clusterRange:
        f.root._v_attrs.kmin = min_ncluster
        f.root._v_attrs.kmax = max_ncluster
    else:
        f.root._v_attrs.kmin = n_clusters
        f.root._v_attrs.kmax = n_clusters
    # save biggest cluster size
    f.root._v_attrs.bgs = bgs
    # save biggest cluster partition
    f.root._v_attrs.bgp = bgp


    if isinstance(file, str):
        f.close()

def saveEnsembleToFileHDF(file, ensemble):
    # open/create file if file is string of path
    if isinstance(file, str):

        abspath = os.path.abspath(file)

        # check for directory existence
        if not os.path.exists(os.path.dirname(abspath)):
            raise IOError("Directory not found:" + os.path.dirname(abspath))

        # create/open file in write mode
        f = tables.openFile(abspath, 'w')
    elif isinstance(file, tables.file.File):
        f = file
    else:
        raise TypeError("file must be string of path or a tables.file.File")

    for i, p in enumerate(ensemble):
        savePartitionToFileHDF(f, p, i)

    # number of samples
    f.root._v_attrs.N = n_samples_from_partition(ensemble[0])
    # biggest cluster size in ensemble
    f.root._v_attrs.bgs = biggest_cluster_size(ensemble)

    # close file if it was opened here
    if isinstance(file, str):
        f.close()

def savePartitionToFileHDF(file, partition, n):
    if isinstance(file, str):

        abspath = os.path.abspath(file)

        # check for directory existence
        if not os.path.exists(os.path.dirname(abspath)):
            raise IOError("Directory not found:" + os.path.dirname(abspath))

        # create/open file in write mode
        f = tables.openFile(abspath, 'w')
    elif isinstance(file, tables.file.File):
        f = file
    else:
        raise TypeError("file must be string of path or a tables.file.File")

    datatype = partition[0].dtype
    atom = tables.Atom.from_dtype(datatype)
    filters = tables.Filters(complib='blosc', complevel=5)

    f.createGroup('/','part{}'.format(n))
    for i,c in enumerate(partition):
        ds = f.createCArray('/part{}'.format(n), 'clust{}'.format(i), atom,
                            c.shape, filters=filters)
        ds[:] = c

    if isinstance(file, str):
        f.close()

def loadEnsembleFromFileHDF(file, generator=False, metadata=False):
    """Loads an ensemble from file. If generator is set to True, then the
    function returns a generator that will read one partition at a time instead
    of the whole ensemble being read to memory at once."""

    if isinstance(file, str):

        abspath = os.path.abspath(file)

        # check for directory existence
        if not os.path.exists(abspath):
            raise IOError("File not found:" + abspath)

        # create/open file in write mode
        f = tables.openFile(abspath, 'r')
    elif isinstance(file, tables.file.File):
        f = file
    else:
        raise TypeError("file must be string of path or a tables.file.File")

    # load metadata
    if metadata:
        md = dict()
        if hasattr(f.root._v_attrs, 'N'):
            md['n_samples'] = f.root._v_attrs.N

        if hasattr(f.root._v_attrs, 'dim'):
            md['n_dims'] = f.root._v_attrs.dim

        if hasattr(f.root._v_attrs, 'kmin'):
            md['kmin'] = f.root._v_attrs.kmin

        if hasattr(f.root._v_attrs, 'kmax'):
            md['kmax'] = f.root._v_attrs.kmax

        if hasattr(f.root._v_attrs, 'bgs'):
            md['biggest cluster size'] = f.root._v_attrs.bgs

    # file should be closed if it was open inside this function
    closeFile = True if isinstance(file, str) else False

    if not generator:
        ensemble = _loadWholeEnsembleFromFileHDF(f, closeFile)
    else:
        ensemble = _loadGeneratorEnsembleFromFileHDF(f, closeFile)

    if metadata:
        returnset = (ensemble, md)
    else:
        returnset = ensemble

    return returnset

def _loadWholeEnsembleFromFileHDF(file, close=False):
    """This function loads the entire ensemble to memory.
    _file_ is a _tables.File.file_.
    _close_ is a _boolean_ that indicates if the file should be closed in the
    end or not.
    """
    ensemble = list()
    for p in file.iterNodes('/'):
        # create list of cluster arrays and append it to ensemble list
        ensemble.append([i[:] for i in file.iterNodes(p)])

    if close:
        file.close()

    return ensemble

def _loadGeneratorEnsembleFromFileHDF(file, close=False):
    """This function is a generator that goes through all the partitions in the
    ensemble file, loading each to memory as needed.
    _file_ is a _tables.File.file_.
    _close_ is a _boolean_ that indicates if the file should be closed in the
    end or not.
    """
    for p in file.iterNodes('/'):
        # create list of cluster arrays and append it to ensemble list
        yield [i[:] for i in file.iterNodes(p)]

    if close:
        file.close()


def loadPartitionFromFileHDF(file, n):
    if isinstance(file, str):

        abspath = os.path.abspath(file)

        # check for directory existence
        if not os.path.exists(abspath):
            raise IOError("File not found:" + abspath)

        # create/open file in write mode
        f = tables.openFile(abspath, 'r')
    elif isinstance(file, tables.file.File):
        f = file
    else:
        raise TypeError("file must be string of path or a tables.file.File")

    node = f.getNode('/part{}'.format(n))
    part = [i[:] for i in f.iterNodes(node)]

    if isinstance(file, str):
        f.close()

    return part

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                        UTIL FUNCTIONS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def n_samples_from_partition(partition):
    return sum(map(len, partition))

def biggest_cluster_size(ensemble):
    return max([max(map(np.size,p)) for p in ensemble])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                        PLOTTING ENSEMBLE
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PlotEnsemble:
    def __init__(self, ensemble, data):
        self.ensemble = ensemble
        self.data = data
        self.curr_partition = 0
        self.n_partitions = len(ensemble)

        self.missing_hulls = 0

    def plot(self, num = None, draw_perimeter=False):
        if num is None:
            self._plotPartition(self.curr_partition, draw_perimeter)
            if self.curr_partition < self.n_partitions - 1:
                self.curr_partition += 1
            else:
                self.curr_partition = 0
        elif num >= self.n_partitions or num < 0:
            raise Exception("Invalid partition index.")
        else:
            self.curr_partition = num
            self._plotPartition(self.curr_partition, draw_perimeter)

    def _plotPartition(self, clust_idx, draw_perimeter=False):

        if not draw_perimeter:
            for clust in self.ensemble[clust_idx]:
                plt.plot(self.data[clust, 0], self.data[clust, 1], '.')

        else:
            from scipy.spatial import ConvexHull
            self.missing_hulls = 0
            for clust in self.ensemble[clust_idx]:
                points = self.data[clust]
                plt.plot(points[:,0], points[:,1], 'o')

                if clust.size < 5:
                    self.missing_hulls += 1
                    continue
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.title("Partition #{}, Num. clusters = {}".format(clust_idx,
            len(self.ensemble[clust_idx])))

    def maxClusterSize(self):
        return biggest_cluster_size(ensemble)
