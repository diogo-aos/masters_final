# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:59:23 2015

@author: Diogo Silva


Generates partitions using K-Means.
Can generate data as well.


# TODO:
- exception handling (file management)
- advanced logging
"""

import numpy as np

from timeit import default_timer as timer # timing
import sys

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    GLOBAL VARS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

synthetic_data = False
nsamples = None
ndims = None
minclusters = None
maxclusters = None

filename_base = None
mode = "numpy"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    ARGUMENTS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="where to read data from",type=str)
parser.add_argument("-n", "--nsamples", help="number of samples for synthetic data",type=int)
parser.add_argument("-D", "--dimension", help="dimension for synthetic data",type=int)
parser.add_argument("-C", "--centers", help="number of clusters for synthetic data",type=int)
parser.add_argument("-i", "--iterations", help="number of iterations",type=int)
parser.add_argument("-m", "--mode", help="computation mode (cuda,numpy,python)",type=str)
parser.add_argument("-s", "--sufix", help="sufix of saved files",type=str)
parser.add_argument("-np", "--npartitions", help="number of partitions",type=int)

#parser.add_argument("-c", "--clusters", help="number of clusters",type=int)
parser.add_argument("-Mc", "--maxclusters", help="maximum number of clusters",type=int)
parser.add_argument("-mc", "--minclusters", help="minimum number of clusters",type=int)

parser.add_argument("-dir", "--directory", help="directory in which to save files",type=str)

args = parser.parse_args()


# data filename
if args.data == "synthetic":
    synthetic_data = True
    filename_base = args.directory + args.sufix
else:
    synthetic_data = False
    data_filename = args.data
    #wo_ext = args.data.split(".csv")[0]
    #filename_base = args.directory + wo_ext + "_" + args.sufix
    filename_base = args.directory + args.sufix


# synthetic data parameters
if synthetic_data:
    nsamples = args.nsamples
    ndims = args.dimension
    centers = args.centers

iters = args.iterations
npartitions = args.npartitions

maxclusters = args.maxclusters
minclusters = args.minclusters

mode = args.mode

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    LOGGING

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import logging

# Status logging
logger = logging.getLogger('status')
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file handler
handler = logging.FileHandler(filename_base + '.log')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

# create a console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(consoleHandler)

logger.info('Start of logging.')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    GETTING DATA

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



if synthetic_data:
    log_str = "nsamples={0},ndims={1},centers={2}".format(nsamples,ndims,centers)
    logger.info('Generating data:' + log_str)

    from sklearn.datasets import make_blobs # generate gaussian mixture

    data, groundTruth = make_blobs(n_samples=nsamples,n_features=ndims,centers=centers,
                                            center_box=(-1000.0,1000.0))
    try:
        np.savetxt(filename_base + "_ground_truth.csv", groundTruth, delimiter=',')
        np.savetxt(filename_base + "_data.csv", data, delimiter=',')
    except:
        logger.info('Error saving data to:' + log_str)
else:
# read data form csv
    logger.info('Reading data: ' + data_filename)
    try:
        data = np.genfromtxt(data_filename,delimiter=',')
    except IOError:
        print "File not found."
        sys.exit()
    except:
        raise
    nsamples,ndims = data.shape

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    GETTING PARTITIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import K_Means3
from K_Means3 import K_Means

def getPartition(k):
    grouper = K_Means()
    grouper._centroid_mode = "index"
    grouper.fit(data, k, iters=iters, mode=mode, cuda_mem='manual',tol=1e-4,max_iters=300)
    return grouper.partition

def savePartitionToFile(partition,filename):
    text_file = open(filename, "w")
    
    for clust in partition:
        cluster_line = ','.join(['{}'.format(num) for num in clust])
        cluster_line += '\n'

        text_file.write(cluster_line)

    text_file.close()

    logger.info('Saved partition: ' + filename)

for p in xrange(npartitions):
    
    nclusters = np.random.randint(minclusters,maxclusters)
    
    log_str = "#{0}, clusters={1}".format(p,nclusters)
    logger.info('Generating partition: ' + log_str)

    partition = getPartition(nclusters)

    #save partition to file
    savePartitionToFile(partition,filename_base + '_partition_' + str(p) + '.csv')