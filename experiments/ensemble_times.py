
# coding: utf-8

import MyML.utils.profiling as myProf

t = myProf.Timer()
t.tic()

import numpy as np
import pandas as pd

import os.path
import sys

import MyML.helper.partition as part
import MyML.cluster.K_Means3 as myKM
import MyML.metrics.accuracy as myAcc
import MyML.EAC.eac_new as myEAC
import MyML.EAC.sparse as mySpEAC
import MyML.EAC.rules as kminRules

# Setup logging
import logging

# for arguments
import argparse

# to explicitely call garbage collection
import gc

# call breakpoints
import pdb

print "Import time:", t.tac()


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="where to read data from",type=str)
parser.add_argument("-ef", "--ensemblefolder",
                    help="where to read/write ensemble", type=str,
                    default=None)
parser.add_argument("-d", "--diskdir",
                    help="where to store the disk graph for mst", 
                    type=str)
parser.add_argument('-Mc', "--max_cardinality", type=int, default=0)
parser.add_argument('-mc', "--min_cardinality", type=int, default=0)
parser.add_argument('-rb', "--reboot", action='store_true',
                    help="reboot from previous state. Must specify previous"\
                         " state with -rp arguments")
parser.add_argument('-rp', "--reboot_parameters", type=str,
                    default=False, nargs=5,
                    help='reboot arguments: [#samples] [rule name] '\
                         '[type of mat] [round] [results index]')
parser.add_argument('-y', "--yes", help="don't ask confirmation of folder",
                    action='store_true')
args = parser.parse_args()

folder = args.folder
diskdir = args.diskdir
ensemble_dir = args.ensemblefolder
max_cardinality = args.max_cardinality
min_cardinality = args.min_cardinality
reboot = args.reboot
reboot_parameters = args.reboot_parameters
if reboot and not reboot_parameters:
    raise ValueError("Reboot parameters are necessary if reboot is True.")

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


if ensemble_dir is None:
    ensemble_dir = folder
else:
    if not os.path.exists(ensemble_dir):
        print "Path does not exist: ", ensemble_dir
        sys.exit(1)
    ensemble_dir = os.path.abspath(ensemble_dir)  

     # check if path is folder
    if not os.path.isdir(ensemble_dir):
        print "Ensemble directory path is not directory: ", ensemble_dir
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

    tri = (n_e - n_s) * (val_s - val_e) / 2.0
    r_rect = (1.0 - n_e) * (val_s - val_e)
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




rules = [kminRules.rule1, kminRules.rule2, kminRules.rule4, kminRules.rule5]
rules_docs = [rule.__doc__ for rule in rules]

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
res_cols = ['n_samples', 'rule', 'kmin', 'kmax',
            't_ensemble', 't_build', 't_sl', 't_accuracy_CI', 't_accuracy_H',
            't_sl_disk', 't_store', 't_accuracy_CI_disk', 't_accuracy_H_disk',
            'biggest_cluster',
            'type_mat',
            'n_assocs', 'n_max_degree',
            'min_degree', 'max_degree', 'mean_degree', 'std_degree',
            'accuracy_CI', 'accuracy_H', 'sl_clusts',
            'accuracy_CI_disk', 'accuracy_H_disk', 'sl_clusts_disk',
            'round', 'disk']

type_mats = ["full",
             "full condensed",
             "sparse complete",
             "sparse condensed const",
             "sparse condensed linear"]
rounds = 5
res_lines = rounds * len(cardinality) * len(rules) * len(type_mats)

results = pd.DataFrame(index=range(res_lines), columns=res_cols)


t = myProf.Timer() # timer

# ensemble properties
n_partitions = 100
n_iters = 3

# EAC properties
sparse_max_assocs_factor = 3

if reboot:
    n_old = int(reboot_parameters[0])
    rule_old = reboot_parameters[1]
    type_mat_old = reboot_parameters[2]
    r_old = int(reboot_parameters[3])
    res_idx_old = int(reboot_parameters[4])

    if rule_old not in rules_docs:
        raise ValueError("Reboot rule does not exist.")

    if type_mat_old not in type_mats:
        raise ValueError("Reboot type mat does not exist.")

# ## run

logger.info("Starting experiment...")

# In[198]:

if reboot:
    res_idx = res_idx_old + 1
else:
    res_idx = 0
    
for d in div: # for each size of dataset   
    # sample data
    data_sampled = np.ascontiguousarray(data[::d])
    gt_sampled = np.ascontiguousarray(gt[::d])
    n = data_sampled.shape[0]

    if reboot and n_old is not None:
        if n_old != n:
            continue
        else:
            n_old = None

    if max_cardinality != False and n >= max_cardinality:
        continue

    if min_cardinality != False and n < min_cardinality:
        continue

    if n > 2500000:
        break

    for rule in rules: # for each kmin rule

        if reboot and rule_old is not None:
            if rule_old != rule.__doc__:
                continue
            else:
                rule_old = None

        n_clusts = rule(n)

        logger.info("* * * * * * * * * * * * * * * * * *")
        logger.info("Num. samples: {}".format(n))
        logger.info("New rule: {}".format(rule.__doc__))
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
        if n >= 1000000:
            lm = 'numba'
        else:
            lm = 'cuda'

        generator = myKM.K_Means(label_mode=lm, cuda_mem="manual")

        logger.info("No ensemble detected. Generating ensemble...")
        t.reset()
        t.tic()
        ensemble = part.generateEnsemble(data_sampled, generator, n_clusts,
                                         n_partitions, n_iters)
        t_ensemble = t.tac()

        results.n_samples[res_idx] = n # n_samples
        results.rule[res_idx] = rule.__doc__ # rule
        results.kmin[res_idx] = n_clusts[0] # kmin
        results.kmax[res_idx] = n_clusts[1] # kmax
        results.t_ensemble[res_idx] = t_ensemble # ensemble time

        results.to_csv(os.path.join(folder, "ensemble_results_kmin.csv"))
        res_idx += 1   