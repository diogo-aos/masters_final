import MyML.utils.profiling as myProf

tImport = myProf.Timer()
tImport.tic()

import numpy as np
import MyML.helper.partition as part
import MyML.cluster.K_Means3 as myKM
import MyML.metrics.accuracy as myAcc
import MyML.EAC.eac_new as myEAC
import MyML.EAC.sparse as mySpEAC

import gc
import argparse
import os.path


def correspond(l0, l1):
    """gets two labels arrays and, if they have the
    same number of clusters, tries to equal the label assignment
    """
    l0_unique = np.unique(l0)
    if l0_unique.size != np.unique(l1).size:
        return -1

    inc = l0_unique.max() + 100

    # increment all labels
    for l in l0_unique:
        l0[l0==l] = l + inc

    # final change
    for l in np.unique(l0):
        first_idx = np.where(l0 == l)[0][0]
        final_val = l1[first_idx]
        l0[l0==l] = final_val

    return 0

def eq_matrix(*args):
    n = len(args)
    mat = np.zeros((n,n), dtype=np.bool)
    for i, a1 in enumerate(args):
        for j, a2 in enumerate(args):
            mat[i,j] = a1 == a2
    return mat

print "import time:", tImport.tac()

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name of experiment",type=str)
parser.add_argument("folder", help="where to read data from",type=str)
parser.add_argument("-d", "--diskdir",
                    help="where to store the disk graph for mst", 
                    type=str)
parser.add_argument("-nd", "--n_samples",
                    help="number of samples to use from data", 
                    type=int)
parser.add_argument('-y', "--yes", help="don't ask confirmation of folder",
                    action='store_true')
args = parser.parse_args()

name = args.name
folder = args.folder
diskdir = args.diskdir
n_samples = args.n_samples

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


# rules for picking kmin kmax 
def rule1(n):
    """sqrt"""
    k = [np.sqrt(n)/2, np.sqrt(n)]
    k = map(np.ceil,k)
    k = map(int, k)
    return k


data = np.load(data_path)
gt = np.load(gt_path)

# sample data
n_data = data.shape[0]
div_data = int(n_data / n_samples)
data_sampled = np.ascontiguousarray(data[::div_data])
gt_sampled = np.ascontiguousarray(gt[::div_data])
n = data_sampled.shape[0]

# ensemble properties
n_partitions = 100
n_iters = 3

# EAC properties
sparse_max_assocs_factor = 3



rule = rule1

t = myProf.Timer()

name = name + '_'
ensemble_filename = os.path.join(folder,name + "ensemble_{}_{}.hdf".format(n, rule.__doc__))
if not os.path.exists(ensemble_filename):
    print "No ensemble detected. Generating ensemble..."
    generator = myKM.K_Means(cuda_mem="manual")
    n_clusts = rule(n)
    t.reset()
    t.tic()
    ensemble = part.generateEnsemble(data_sampled, generator, n_clusts,
                                     n_partitions, n_iters)
    t.tac()
    part.saveEnsembleToFileHDF(ensemble_filename, ensemble)
    print "Saved ensemble in file: {}".format(ensemble_filename)
    t_ensemble = t.elapsed
else:
    print "Ensemble detected in file {}. Loading ensemble...".format(ensemble_filename)
    ensemble = part.loadEnsembleFromFileHDF(ensemble_filename)
    t_ensemble = -1

print "ensemble time:", t_ensemble

max_cluster_size = part.biggest_cluster_size(ensemble)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
str_desc = "full condensed"
print str_desc + " building matrix..."
eacEst1 = myEAC.EAC(n_samples=n, sparse=False, condensed=True, n_partitions=n_partitions)
t.reset()
t.tic()
eacEst1.buildMatrix(ensemble)
t.tac()
print str_desc + " build time:", t.elapsed

print str_desc + " SL..."
t.reset()
t.tic()
labels_fc = eacEst1.finalClustering(n_clusters=0)
t.tac()
print str_desc + " SL time: {}".format(t.elapsed)

labels_filename = name + "labels_{}_{}_{}.npy".format(n, rule.__doc__, "fc")
np.save(os.path.join(folder,labels_filename), labels_fc)

t.reset()   
t.tic()
print str_desc + " scoring (consistency)..."
accEst1_1 = myAcc.ConsistencyIndex(n)
accEst1_1.score(gt_sampled, labels_fc)

print str_desc + " scoring (Hungarian)..."
accEst1_2 = myAcc.HungarianIndex(n)
accEst1_2.score(gt_sampled, labels_fc)

t.tac()

print str_desc + " accuracy(consistency): {}".format(accEst1_1.accuracy)
print str_desc + " accuracy(Hungarian): {}".format(accEst1_2.accuracy)

print str_desc + " accuracy time: {}".format(t.elapsed)

gc.collect()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
str_desc = "sparse const condensed"
print str_desc + " building matrix..."
eacEst2 = myEAC.EAC(n_samples=n, sparse=True, condensed=True, n_partitions=n_partitions, sparse_keep_degree=True)
t.reset()
t.tic()
eacEst2.buildMatrix(ensemble)
t.tac()
print str_desc + " build time:", t.elapsed

coassoc_store_path = os.path.join(diskdir, name + "coassoc.h5")

t.reset()
t.tic()
eacEst2.coassoc.store(coassoc_store_path, delete=False,
                      indptr_expanded=True,
                      store_degree=True)
t.tac()
print str_desc + " store time:", t.elapsed

print str_desc + " SL..."
t.reset()
t.tic()
labels_scc = eacEst2.finalClustering(n_clusters=0)
t.tac()
print str_desc + " SL time: {}".format(t.elapsed)

labels_filename = name + "labels_{}_{}_{}.npy".format(n, rule.__doc__, "scc")
np.save(os.path.join(folder,labels_filename), labels_scc)

t.reset()
t.tic()
print str_desc + " scoring (consistency)..."
accEst2_1 = myAcc.ConsistencyIndex(n)
accEst2_1.score(gt_sampled, labels_scc)

print str_desc + " scoring (Hungarian)..."
accEst2_2 = myAcc.HungarianIndex(n)
accEst2_2.score(gt_sampled, labels_scc)

t.tac()

print str_desc + " accuracy (consistency): {}".format(accEst2_1.accuracy)
print str_desc + " accuracy (Hungarian): {}".format(accEst2_2.accuracy)
print str_desc + " accuracy time: {}".format(t.elapsed)

gc.collect()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# str_desc = "sparse const condensed disk"
# print str_desc + " building matrix..."
# eacEst3 = myEAC.EAC(n_samples=n, sparse=True, condensed=True, n_partitions=n_partitions,
#                     sl_disk=True, sl_disk_dir=diskdir,
#                     coassoc_store_path=os.path.join(diskdir, name + "coassoc.h5"))
# t.reset()
# t.tic()
# eacEst3.buildMatrix(ensemble)
# t.tac()
# print str_desc + " build time:", t.elapsed

# print str_desc + " SL..."
# t.reset()
# t.tic()
# labels_scc_disk = eacEst3.finalClustering(n_clusters=0)
# t.tac()
# print str_desc + " SL time: {}".format(t.elapsed)

t.reset()
t.tic()
nclusts, labels_scc_disk = myEAC.sp_sl_lifetime_disk(coassoc_store_path,
                                                     n_partitions,
                                                     n_clusters=8,
                                                     index_dir=diskdir)
t.tac()
print str_desc + " SL time: {}".format(t.elapsed)

labels_filename = name + "labels_{}_{}_{}.npy".format(n, rule.__doc__,
                                                      "scc_disk")
np.save(os.path.join(folder,labels_filename), labels_scc_disk)

t.reset()
t.tic()
print str_desc + " scoring (consistency)..."
accEst3_1 = myAcc.ConsistencyIndex(n)
accEst3_1.score(gt_sampled, labels_scc_disk)

print str_desc + " scoring (Hungarian)..."
accEst3_2 = myAcc.HungarianIndex(n)
accEst3_2.score(gt_sampled, labels_scc_disk)

t.tac()

print str_desc + " accuracy (consistency): {}".format(accEst3_1.accuracy)
print str_desc + " accuracy (Hungarian): {}".format(accEst3_2.accuracy)
print str_desc + " accuracy time: {}".format(t.elapsed)

gc.collect()


print "fc CI, fc H, scc CI, scc H, scc disk CI, scc disk H"
print (accEst1_1.accuracy, accEst1_2.accuracy,
       accEst2_1.accuracy, accEst2_2.accuracy,
       accEst3_1.accuracy, accEst3_2.accuracy)
print eq_matrix(accEst1_1.accuracy, accEst1_2.accuracy,
                accEst2_1.accuracy, accEst2_2.accuracy,
                accEst3_1.accuracy, accEst3_2.accuracy)

print np.all(labels_scc == labels_fc)


# experiments/study_kmin.py -f datasets/gauss10e6_overlap -d /media/Data/diogoaos_tmp/ -Mc 5000 -mc 1000
# results[['accuracy_CI', 't_accuracy_CI', 'type_mat', 'rule']][:res_idx]