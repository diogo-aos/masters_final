# -*- coding: utf-8 -*-
"""
Created on 26-04-2015

@author: Diogo Silva

Analysis on the influence of the number of prototypes K.

# TODO:
- make it work for EAC with KNN and K methods
- validate arguments
"""

import numpy as np
import eac
import K_Means3
from timeit import default_timer as timer # timing
import determine_ci

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    ARGUMENTS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-P", "--partitions", help="file with filenames of partitions",type=str)
parser.add_argument("-G", "--ground_truth", help="filename of file with ground truth",type=str)
parser.add_argument("-K", "--nprots", help="filename with number of prototypes to use; can have more than one number",type=str) #used only with knn and k
parser.add_argument("-r", "--rounds", help="number of rounds to perform for each number of prototypes",type=int)

parser.add_argument("-m", "--mode", help="computation mode (cuda,numpy,python)",type=str)
parser.add_argument("-C", "--clusters", help="number of clusters to use in clustering",type=int)
parser.add_argument("-M", "--maxiter", help="maximum iterations to use on K-Means",type=int)
parser.add_argument("-t", "--tolerance", help="tolerance to use for K-Means stopping condition",type=float)

parser.add_argument("-a", "--coassoc", help="coassociation method (random,knn,k)",type=str)
parser.add_argument("-d", "--data", help="filename of data",type=str) #used only with knn and k
parser.add_argument("-p", "--prefix", help="prefix to put on output files",type=str) #used only with knn and k

args = parser.parse_args()


# read partition file names to a list
partition_list_file = args.partitions
partition_files = list()
with open(partition_list_file,"r") as pfile:
	for line in pfile:
		partition_files.append(line.rstrip('\n'))

# read ground truth
ground_truth = np.genfromtxt(args.ground_truth,dtype=np.int32)

# read number of prototypes to use
K = np.genfromtxt(args.nprots,dtype=np.int32)

# read number of clusters to use in K-Means
nclusters = args.clusters

# get tolerance to use in k-means
tol=args.tolerance

# get maximum number of iterations to allow in k-means
max_iters = args.maxiter

# get EAC method
eac_method = args.coassoc

# read K-Means mode
kmeans_mode = args.mode

# get number of rounds
rounds = args.rounds


# read filename


# read prefix
prefix = args.prefix

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
handler = logging.FileHandler(prefix + '_log.log')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

# create a console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(consoleHandler)

# Results logging
resultsLogger = logging.getLogger('results')
resultsLogger.setLevel(logging.INFO)

# create a file handler for results
resultsHandler = logging.FileHandler(prefix + '_results.csv')
resultsHandler.setLevel(logging.INFO)

resultsLogger.addHandler(resultsHandler)

# output format
# index 			: simple index column, 0, ..., X
# nprots 			: number of prototypes used
# nclusters 		: number of clusters used
# iters 			: number of iterations K-Means performed
# pc_idx 			: consistency index
# time 				: various times
# 	eac 			: time that EAC took
# 	k_means 		: time that K-Means took
# 	pcidx 			: time that consistency index took

resultsLogger.info('index,nprots,nclusters,iters,pc_idx,time_eac,time_k_means,time_pcidx') #csv header
logger.info('Start of logging.')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    ENV VARIABLES

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# eac_method 						: method to use in EAC prototypes choice (random, KNN or K centroids)
# kmeans_mode 						: computation mode to use in K-Means (cuda, numpy or python)
# partition_files 					: list of files with the partitions
# K 								: list with number of prototypes to use
# nclusters 						: number of clusters to use in K-Means
# tol 								: tolerance to use in K-Means stop condition
# maxiters 							: maximum number of iterations allowed in K-Means
# rounds 							: number of rounds for each number of prototypes
nsamples = ground_truth.size #  	: number of samples in dataset
index = 0 # 						: index of test iteration


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    COMPUTE

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

for nprots in K:
	for r in xrange(rounds):
		#
		## compute EAC
		#
		log_line = "Starting EAC: method={}, prototypes={}".format(eac_method,nprots)
		logger.info(log_line)


		start = timer()
		estimator=eac.EAC(nsamples)
		estimator.fit(partition_files,files=True,assoc_mode='prot', prot_mode=eac_method, nprot=nprots,build_only=True)
		time_eac = timer() - start

		log_line = "EAC done: time={}s.".format(time_eac)
		logger.info(log_line)


		#
		## K-Means
		#

		log_line = "Starting K-Means: method={}, cluster={}".format(kmeans_mode,nclusters)
		logger.info(log_line)

		grouper = K_Means3.K_Means()
		grouper._centroid_mode = "index"
		grouper.fit(estimator._coassoc, nclusters, iters="converge", mode=kmeans_mode, cuda_mem='manual',tol=1e-4,max_iters=300)
		time_kmeans = timer() - start

		log_line = "K-Means done: time={}s, iterations={}".format(time_kmeans,grouper.iters_)
		logger.info(log_line)

		#
		## Consistency index
		#

		log_line = "Starting consistency index"
		logger.info(log_line)

		start = timer()
		ci=determine_ci.ConsistencyIndex(N=nsamples)
		accuracy=ci.score(ground_truth,grouper.labels_,format='array')
		time_pcidx = timer() - start

		log_line = "Consistency index done: accuracy={},time={}s".format(accuracy,time_pcidx)
		logger.info(log_line)

	    # save results
		result_line = "{0},{1},{2},{3},{4},{5},{6},{7}".format(index,nprots,nclusters,grouper.iters_,accuracy,time_eac,time_kmeans,time_pcidx)
		resultsLogger.info(result_line)

		# delete big structures
		del estimator, grouper, ci

		index += 1