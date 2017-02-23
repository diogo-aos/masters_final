
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd


# In[2]:

import MyML.helper.partition as part
import MyML.cluster.eac as eac
import MyML.cluster.K_Means3 as myKM
import MyML.metrics.accuracy as accuracy
import MyML.utils.profiling as myProf

# Setup logging
import logging

# Status logging
logger = logging.getLogger('status')
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file handler
handler = logging.FileHandler('study_kmin.log')
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
    k = [n *1.0 / sk, th * n * 1.0 / sk]
    k = map(np.ceil,k)
    k = map(int, k)
    return k

def rule4(n):
    """sk=sqrt/2,th=30%"""
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

folder = "/home/diogoaos/QCThesis/datasets/gauss1e6/"


# In[ ]:

logger.info("Loading dataset...")

data = np.genfromtxt(folder + "data.csv", delimiter=',', dtype=np.float32)
gt = np.genfromtxt(folder + "gt.csv", delimiter=',', dtype=np.int32)


# In[120]:

mem_full_max = 20 * 2**30 # max mem full mat can take

cardinality = [500,1e3,5e3,1e4,2.5e4,5e4,1e5,2.5e5,5e5,1e6,2.5e6]
cardinality = map(int,cardinality)

total_n = data.shape[0]
div = map(lambda n: total_n/n,cardinality)

rounds = 5
res_lines = rounds * len(cardinality) * len(rules)
res_cols = ['n_samples', 'rule', 'kmin', 'kmax', 't_ensemble', 'type_mat',
            't_build', 'n_assocs', 'max_assoc', 't_sl', 'accuracy', 'round']
results = pd.DataFrame(index=range(res_lines),columns=res_cols)

t = myProf.Timer() # timer

# ensemble properties
n_partitions = 100
n_iters = 3

# EAC properties
assoc_mode = "full"
prot_mode = "none"


# ## run

logger.info("Starting experiment...")

# In[198]:

res_idx = 0
for d in div: # for each size of dataset
   
    # sample data
    data_sampled = np.ascontiguousarray(data[::d])
    #gt_sampled = gt[::d]
    n = data_sampled.shape[0]

    logger.info("Sampled of {} patterns.".format(n))
    
    # pick sparse on full matrix
    if n **2 < mem_full_max:
        mat_sparse = False
    else:
        mat_sparse = True
    
    for rule in rules: # for each kmin rule
        n_clusts = rule(n)

        logger.info("Rule: {}".format(rule.__doc__))
        logger.info("kmin: {}, kmax: {}".format(n_clusts[0], n_clusts[1]))

        for r in range(rounds): # for each round
            logger.info("Round: {}".format(r))

            results.round[res_idx] = r # round number
            results.n_samples[res_idx] = n # n_samples
            results.rule[res_idx] = rule.__doc__ # rule
            results.kmin[res_idx] = n_clusts[0] # kmin
            results.kmax[res_idx] = n_clusts[1] # kmax
            results.type_mat[res_idx] = mat_sparse # type of matrix
    
            logger.info("Generating ensemble...")

            generator = myKM.K_Means(cuda_mem="manual")
            
            t.tic()
            ensemble = part.generateEnsemble(data_sampled, generator, n_clusts, n_partitions, n_iters)
            t.tac()
            
            results.t_ensemble[res_idx] = t.elapsed # ensemble time
            
            logger.info("Sparse matrix: {}".format(mat_sparse))
            logger.info("Building matrix...")

            myEst = eac.EAC(n, mat_sparse=mat_sparse)
            
            t.tic()
            myEst.fit(ensemble, files=False, assoc_mode=assoc_mode, prot_mode=prot_mode)
            t.tac()
            
            results.t_build[res_idx] = t.elapsed # build time
            results.n_assocs[res_idx] = myEst.getNNZAssocs() # number of associations
            results.max_assoc[res_idx] = myEst.getMaxAssocs() # max number of association in a sample
            
            if mat_sparse: # don't do SL if sparse matrix -> NOT IMPLEMENTED
                results.to_csv(folder + "results_kmin.csv")
                res_idx += 1
                del generator, ensemble, myEst
                continue

            # logger.info("SL clustering...")

            # t.tic()
            # labels = myEst._lifetime_clustering()
            # t.tac()
            
            # logger.info("Scoring accuracy...")
            # accEst = accuracy.HungarianIndex(n)
            # accEst.score(gt_sampled, labels)
            
            # results.t_sl[res_idx] = t.elapsed # build time
            # results.accuracy[res_idx] = accEst.accuracy
            
            results.to_csv(folder + "results_kmin.csv")
            res_idx += 1

            del generator, ensemble, myEst#, accEst
    del data_sampled#, gt_sampled