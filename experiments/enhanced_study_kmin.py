
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

import os.path
import sys

# In[2]:

import MyML.helper.partition as part
import MyML.cluster.K_Means3 as myKM
import MyML.metrics.accuracy as myAcc
import MyML.utils.profiling as myProf
import MyML.EAC.eac_new as myEAC
import MyML.EAC.sparse as mySpEAC

# Setup logging
import logging

# for arguments
import argparse

# to explicitely call garbage collection
import gc

# call breakpoints
import pdb



parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="where to read data from",type=str)
parser.add_argument("-d", "--diskdir",
                    help="where to store the disk graph for mst", 
                    type=str)
parser.add_argument('-y', "--yes", help="don't ask confirmation of folder",
                    action='store_true')
args = parser.parse_args()

folder = args.folder
diskdir = args.diskdir

## path and files checks
# check if path exists
if not os.path.exists(folder):
    print "Path does not exist: ", folder
    sys.exit(1)
folder = os.path.abspath(folder)

# check if path is folder
if not os.path.isdir(folder):
    print "Path is not directory: ", folder
    sys.exit(1)

# check if there is a data file
not_data_or_gt = False
data_path = os.path.join(folder,"data.npy")
if not os.path.exists(data_path):
    print "Directory doesn't have a data file: ", data_path
    not_data_or_gt = True

# check if there is a ground truth file
gt_path = os.path.join(folder,"gt.npy")
if not os.path.exists(gt_path):
    print "Directory doesn't have a ground truth file: ", gt_path
    not_data_or_gt = True

if not_data_or_gt:
    sys.exit(1)

if not os.path.exists(diskdir):
    print "Disk directory path does not exist: ", diskdir
    sys.exit(1)
diskdir = os.path.abspath(diskdir)

# check if path is folder
if not os.path.isdir(diskdir):
    print "Disk directory path is not directory: ", folder
    sys.exit(1)

# await confirmation
if not args.yes:
    raw_input("Folder: {}\nIs this correct?".format(folder))
else:
    print "Folder being used is: {}".format(folder)

# await confirmation for disk directory
if not args.yes:
    raw_input("Disk directory: {}\nIs this correct?".format(diskdir))
else:
    print "Folder being used is: {}".format(diskdir)

# Status logging
logger = logging.getLogger('status')
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file handler
handler = logging.FileHandler(os.path.join(folder,'study_kmin.log'))
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

# create a console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(consoleHandler)

# # bulk study


## memory functions helper

def sp_lin_mem(n_s, n_e, val_s, val_e):

    tri = (n_e-n_s) * (val_s-val_e) / 2.0
    r_rect = (1-n_e) * (val_s-val_e)
    cut_area = tri + r_rect

    return 1 - cut_area

def compute_mems(n, ma, n_s, n_e, val_s, val_e):
    # full complete
    full_comp = n ** 2

    # full condensed    
    full_cond = n * (n - 1) // 2

    # sparse constant
    sp_const = n * ma * (1 + 4) + 12 * (n + 1) # data, indices, indptr, degree
    sp_const_mst = n * ma * (8 + 4) # argsort and row_ind

    lin_factor = sp_lin_mem(n_s, n_e, val_s, val_e)
    sp_lin = n * ma * lin_factor * (1 + 4) + 12 * (n + 1) # data, indices, indptr, degree
    sp_lin_mst = n * ma * lin_factor * (4 + 8) # argsort and row_ind

    return full_comp, full_cond, sp_const, sp_lin, sp_const_mst, sp_lin_mst

# ## rules

# In[119]:

# rules for picking kmin kmax 
def rule1(n):
    """sqrt"""
    k = [np.sqrt(n)/2, np.sqrt(n)]
    k = map(np.ceil,k)
    k = map(int, k)
    return k

def rule2(n):
    """2sqrt"""
    k =  map(lambda x:x*2,rule1(n))
    return k

def rule3(n, sk, th):
    """fixed s/k"""
    k = [n * 1.0 / sk, th * n * 1.0 / sk]
    k = map(np.ceil,k)
    k = map(int, k)
    return k

def rule4(n):
    """sk=sqrt_2,th=30%"""
    return rule3(n, sk1(n), 1.3)

def rule5(n):
    """sk=300,th=30%"""
    return rule3(n,300, 1.3)

# rules for picking number of samples per cluster
def sk1(n):
    """sqrt/2"""
    return int(np.sqrt(n) / 2)
    
rules = [rule1, rule2, rule4, rule5]


# ## set-up

# In[212]:

logger.info("Loading dataset...")

# data = np.genfromtxt(folder + "data.csv", delimiter=',', dtype=np.float32)
# gt = np.genfromtxt(folder + "gt.csv", delimiter=',', dtype=np.int32)
data = np.load(data_path)
gt = np.load(gt_path)

# In[120]:

mem_full_max = 28 * 2**30 # max mem full mat can take in bytes, 2**30 = 1GB

# number of samples
cardinality = [1e2,2.5e2,5e2,7.5e2,
               1e3,2.5e3,5e3,7.5e3,
               1e4,2.5e4,5e4,7.5e4,
               1e5,2.5e5,5e5,7.5e5,
               1e6,2.5e6]
cardinality = map(int,cardinality)

total_n = data.shape[0]
div = map(lambda n: total_n / n, cardinality)

# prepare results datastructure
res_cols = ['n_samples',
            'rule',
            'kmin', 'kmax',
            't_ensemble', 't_build', 't_sl', 't_accuracy',
            'biggest_cluster',
            'type_mat',
            'n_assocs', 'n_max_degree',
            'min_degree', 'max_degree', 'mean_degree', 'std_degree',
            'accuracy', 'sl_clusts',
            'round','disk']

type_mats = ["full",
             "full condensed",
             "sparse complete",
             "sparse condensed const",
             "sparse condensed linear"]
rounds = 1
res_lines = rounds * len(cardinality) * len(rules) * len(type_mats)

results = pd.DataFrame(index=range(res_lines), columns=res_cols)


t = myProf.Timer() # timer

# ensemble properties
n_partitions = 100
n_iters = 3

# EAC properties
sparse_max_assocs_factor = 3

# ## run

logger.info("Starting experiment...")

# In[198]:

res_idx = 0
for d in div: # for each size of dataset
   
    # sample data
    data_sampled = np.ascontiguousarray(data[::d])
    gt_sampled = np.ascontiguousarray(gt[::d])
    n = data_sampled.shape[0]

    for rule in rules: # for each kmin rule

        if n < 500000 or (n == 500000 and rule.__doc__ in [rule1.__doc__]):
            continue

        n_clusts = rule(n)

        logger.info("* * * * * * * * * * * * * * * * * *")
        logger.info("Num. samples: {}".format(n))
        logger.info("New config: {}".format(rule.__doc__))
        logger.info("* * * * * * * * * * * * * * * * * *")

        # skip if number of clusters is bigger than number of samples
        if n_clusts[1] >= n:
            logger.info("Kmax too large for dataset size. Skipping...")
            continue
        if n_clusts[0] <= 1:
            logger.info("Kmin too little. Skipping...")
            continue            


        ## generate ensemble
        logger.info("Checking for ensemble in folder...")

        generator = myKM.K_Means(cuda_mem="manual")    

        # if there is an ensemble file load it, otherwise generate and save
        ensemble_filename = os.path.join(folder,"ensemble_{}_{}.hdf".format(n, rule.__doc__))
        if not os.path.exists(ensemble_filename):
            logger.info("No ensemble detected. Generating ensemble...")
            t.reset()
            t.tic()
            ensemble = part.generateEnsemble(data_sampled, generator, n_clusts,
                                             n_partitions, n_iters)
            t.tac()
            part.saveEnsembleToFileHDF(ensemble_filename, ensemble)
            logger.info("Saved ensemble in file: {}".format(ensemble_filename))
            t_ensemble = t.elapsed
        else:
            logger.info("Ensemble detected in file {}. Loading ensemble...".format(ensemble_filename))
            ensemble = part.loadEnsembleFromFileHDF(ensemble_filename)
            t_ensemble = -1

        # ensemble_name = "ensemble_" + rule.__doc__ + ".hdf"
        # part.saveEnsembleToFileHDF(os.path.join(folder, ensemble_name), ensemble)

        max_cluster_size = myEAC.biggest_cluster_size(ensemble)

        logger.info("Maximum cluster size: {}".format(max_cluster_size))

        # # # # # # # # # # # # # #
        # check memory usage for different matrix schemes

        # compute memory usage for each type of matrix
        # linear properties for condensed sparse matrix
        n_s = 0.05
        n_e = 1.0
        val_s = 1.0
        val_e = 0.05

        ma = max_cluster_size * sparse_max_assocs_factor

        mems = compute_mems(n, ma, n_s, n_e, val_s, val_e)

        f_mat = mems[0] # full matrix
        fc_mat = mems[1] # full condensed matrix
        sp_const = mems[2] # sparse constant matrix
        sp_lin = mems[3] # sparse linear matrix

        sp_const_mst = mems[4]
        sp_lin_mst = mems[5]

        # # # # # # # # # # # # # # #

        for tm in xrange(len(type_mats)): # for each type of matrix

            for r in range(rounds): # for each round

                if n==500000 and tm <= 1 and rule.__doc__ == rule1.__doc__:
                    continue




    del data_sampled, gt_sampled
    # end of dataset cycle

def full_experiment():
    for 


def get_ensemble(data_sampled, rule):
    n_clusts = rule(n)

    logger.info("* * * * * * * * * * * * * * * * * *")
    logger.info("Num. samples: {}".format(n))
    logger.info("New config: {}".format(rule.__doc__))
    logger.info("* * * * * * * * * * * * * * * * * *")

    # skip if number of clusters is bigger than number of samples
    if n_clusts[1] >= n:
        logger.info("Kmax too large for dataset size. Skipping...")
        continue
    if n_clusts[0] <= 1:
        logger.info("Kmin too little. Skipping...")
        continue            


    ## generate ensemble
    logger.info("Checking for ensemble in folder...")

    generator = myKM.K_Means(cuda_mem="manual")    

    # if there is an ensemble file load it, otherwise generate and save
    ensemble_filename = os.path.join(folder,"ensemble_{}_{}.hdf".format(n, rule.__doc__))
    if not os.path.exists(ensemble_filename):
        logger.info("No ensemble detected. Generating ensemble...")
        t.reset()
        t.tic()
        ensemble = part.generateEnsemble(data_sampled, generator, n_clusts,
                                         n_partitions, n_iters)
        t.tac()
        part.saveEnsembleToFileHDF(ensemble_filename, ensemble)
        logger.info("Saved ensemble in file: {}".format(ensemble_filename))
        t_ensemble = t.elapsed
    else:
        logger.info("Ensemble detected in file {}. Loading ensemble...".format(ensemble_filename))
        ensemble = part.loadEnsembleFromFileHDF(ensemble_filename)
        t_ensemble = -1

    # ensemble_name = "ensemble_" + rule.__doc__ + ".hdf"
    # part.saveEnsembleToFileHDF(os.path.join(folder, ensemble_name), ensemble)

    max_cluster_size = myEAC.biggest_cluster_size(ensemble)

    logger.info("Maximum cluster size: {}".format(max_cluster_size))

    # # # # # # # # # # # # # #
    # check memory usage for different matrix schemes

    # compute memory usage for each type of matrix
    # linear properties for condensed sparse matrix
    n_s = 0.05
    n_e = 1.0
    val_s = 1.0
    val_e = 0.05

    ma = max_cluster_size * sparse_max_assocs_factor

    mems = compute_mems(n, ma, n_s, n_e, val_s, val_e)

    f_mat = mems[0] # full matrix
    fc_mat = mems[1] # full condensed matrix
    sp_const = mems[2] # sparse constant matrix
    sp_lin = mems[3] # sparse linear matrix

    sp_const_mst = mems[4]
    sp_lin_mst = mems[5]

def execute_round(ensemble, r, n, rule, n_clusts):

    logger.info("- - - - - - - - - - - - - - - - - -")
    logger.info("Sampled of {} patterns.".format(n))
    logger.info("Rule: {}".format(rule.__doc__))
    logger.info("kmin: {}, kmax: {}".format(n_clusts[0], n_clusts[1]))
    logger.info("Type of mat: {}".format(type_mats[tm]))
    logger.info("Type of mat tm: {}".format(tm))
    logger.info("Round: {}".format(r))
    logger.info("Building matrix...")
    logger.info("Estimated req. mem. (MB):{}".format(mems[tm] / (1024.0**2)))

    useDiskMST = False

    if tm == 0: # full
        if f_mat > mem_full_max:
            logger.info("not enough memory")
            break
        eacEst = myEAC.EAC(n_samples=n, sparse=False, condensed=False)
        t.reset()
        t.tic()
        eacEst.buildMatrix(ensemble)
        t.tac()

        eacEst.coassoc.getDegree()
        degree = eacEst.coassoc.degree
        nnz = eacEst.coassoc.nnz

        n_max_degree = -1

    elif tm == 1: # full condensed
        if fc_mat > mem_full_max:
            logger.info("not enough memory")
            break
        eacEst = myEAC.EAC(n_samples=n, sparse=False, condensed=True)
        t.reset()
        t.tic()
        eacEst.buildMatrix(ensemble)
        t.tac()

        eacEst.coassoc.getDegree()
        degree = eacEst.coassoc.degree
        nnz = eacEst.coassoc.nnz

        n_max_degree = -1

    elif tm == 2: # sparse complete
        if sp_const > mem_full_max:
            logger.info("not enough memory")
            break

        if sp_const + sp_const_mst > mem_full_max:
            useDiskMST = True

        eacEst = myEAC.EAC(n_samples=n, sparse=True, condensed=False,
                           sparse_keep_degree=True, sl_disk=useDiskMST)
        eacEst.sp_max_assocs_mode="constant"
        eacEst.disk_dir = diskdir

        t.reset()
        t.tic()                    
        eacEst.buildMatrix(ensemble)
        t.tac()

        degree = eacEst.coassoc.degree[:-1]
        nnz = eacEst.coassoc.nnz

        n_max_degree = (degree == ma).sum()

    elif tm == 3: # sparse condensed const
        if sp_const > mem_full_max:
            logger.info("not enough memory")
            break

        if sp_const + sp_const_mst > mem_full_max:
            useDiskMST = True

        eacEst = myEAC.EAC(n_samples=n, sparse=True, condensed=True,
                           sparse_keep_degree=True, sl_disk=useDiskMST)
        eacEst.sp_max_assocs_mode="constant"
        eacEst.disk_dir = diskdir
        t.reset()
        t.tic()                    
        eacEst.buildMatrix(ensemble)
        t.tac()

        degree = eacEst.coassoc.degree[:-1]
        nnz = eacEst.coassoc.nnz

        n_max_degree = (degree == ma).sum()

    elif tm == 4: # sparse condensed linear
        if sp_lin > mem_full_max:
            logger.info("not enough memory")
            break

        if sp_lin + sp_lin_mst > mem_full_max:
            useDiskMST = True
        eacEst = myEAC.EAC(n_samples=n, sparse=True, condensed=True,
                           sparse_keep_degree=True, sl_disk=useDiskMST)
        eacEst.sp_max_assocs_mode="linear"
        eacEst.disk_dir = diskdir
        t.reset()
        t.tic()                    
        eacEst.buildMatrix(ensemble)
        t.tac()

        degree = eacEst.coassoc.degree[:-1]
        nnz = eacEst.coassoc.nnz

        indptr = mySpEAC.indptr_linear(n,
                                       eacEst.sp_max_assocs,
                                        n_s, n_e, val_s, val_e)
        max_degree = indptr[1:] - indptr[:-1]
        n_max_degree = (degree == max_degree).sum()

    else:
        raise NotImplementedError("mat type {} not implemented".format(type_mats[tm]))

    logger.info("Build time: {}".format(t.elapsed))

    results.round[res_idx] = r # round number
    results.n_samples[res_idx] = n # n_samples
    results.rule[res_idx] = rule.__doc__ # rule
    results.kmin[res_idx] = n_clusts[0] # kmin
    results.kmax[res_idx] = n_clusts[1] # kmax
    results.t_build[res_idx] = t.elapsed
    results.type_mat[res_idx] = type_mats[tm] # type of matrix
    results.disk[res_idx] = useDiskMST # SL disk

    results.t_ensemble[res_idx] = t.elapsed # ensemble time
    results.biggest_cluster[res_idx] = max_cluster_size # biggest_cluster

    # number of associations
    results.n_assocs[res_idx] = nnz

    # stats number associations
    results.max_degree[res_idx] = degree.max()
    results.min_degree[res_idx] = degree.min()
    results.mean_degree[res_idx] = degree.mean()
    results.std_degree[res_idx] = degree.std()
    results.n_max_degree[res_idx] = n_max_degree

    logger.info("SL clustering...")

    t.reset()
    t.tic()
    labels = eacEst.finalClustering(n_clusters=0)
    t.tac()
    logger.info("Clustering time: {}".format(t.elapsed))

    labels_filename = "labels_{}_{}_{}".format(n, rule.__doc__, r)
    np.save("")

    results.t_sl[res_idx] = t.elapsed # build time
    results.sl_clusts[res_idx] = eacEst.n_fclusts

    t.reset()
    t.tic()
    # logger.info("Scoring accuracy (consistency)...")
    # accEst = myAcc.ConsistencyIndex(n)
    # accEst.score(gt_sampled, labels)

    logger.info("Scoring accuracy (Hungarian)...")
    accEst = myAcc.HungarianIndex(n)
    accEst.score(gt_sampled, labels)
    
    t.tac()

    logger.info("Accuracy time: {}".format(t.elapsed))

    results.t_accuracy[res_idx] = t.elapsed # accuracy time
    results.accuracy[res_idx] = accEst.accuracy

    # if the accuracy is zero of different from the last round
    # then I want to check what's going on
    if accEst.accuracy == 0 or accEst.accuracy != results.accuracy[res_idx-1]:
        print "breakpoint"
    
    results.to_csv(os.path.join(folder,"results_kmin.csv"))
    res_idx += 1

    del eacEst, accEst, degree, labels
    gc.collect()
    # end of inner most loop