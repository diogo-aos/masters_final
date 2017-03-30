import numpy as np
import tables
import numba as nb
import MyML.utils.profiling as myProf
import MyML.helper.partition as part
import MyML.EAC.sparse as eacSp
import MyML.EAC.eac_new as myEAC
import shutil

t = myProf.Timer()

# ensemble_file = '/home/diogoaos/QCThesis/datasets/gauss10e6_overlap/ensemble_500k_test2.h5'
ensemble_file = '/media/Data/diogoaos_tmp/gaussseparated_ensembles/ensemble_500000_2sqrt.hdf'

coassc_path_ssd = '/home/diogoaos/QCThesis/coassoc.h5'
index_path_ssd = '/home/diogoaos/QCThesis/'

coassc_path_spin = '/media/Data/diogoaos_tmp/coassoc.h5'
index_path_spin = '/media/Data/diogoaos_tmp/'

print "loading ensemble"
t.reset()
t.tic()
ensemble = part.loadEnsembleFromFileHDF(ensemble_file)
print 'load ensemble time: {}'.format(t.tac())

n_samples = part.n_samples_from_partition(ensemble[0])
n_partitions = len(ensemble)
print "number of samples: {}".format(n_samples)
print "number of partitions: {}".format(n_partitions)

ma = eacSp._compute_max_assocs_from_ensemble(ensemble)
ma *= 3
ma = int(ma)

print "memory required: {} MB".format(ma * n_samples * 5 / (1024.0 ** 2))

mat = eacSp.EAC_CSR(n_samples = n_samples, max_assocs=ma, condensed=True, sort_mode='surgical')

print "starting build..."
t = myProf.Timer()
t.reset()
t.tic()
mat.update_ensemble(ensemble)
buildTime = t.tac()
print 'build time: {}'.format(buildTime)

print 'making it diassoc'
mat._condense()
mat.make_diassoc()

print "storing..."
t.reset()
t.tic()
mat.store(coassc_path_spin, indptr_expanded=True, store_degree=False)
print "store time: {}".format(t.tac())

print 'copying file to ssd...'
t.reset()
t.tic()
shutil.copyfile(coassc_path_spin, coassc_path_ssd)
print "copy time: {}".format(t.tac())


print 'SSD clustering...'
t.reset()
t.tic()
nclusts_ssd, labels_ssd = myEAC.sp_sl_lifetime_disk(coassc_path_ssd,
                                                     n_partitions,
                                                     n_clusters=0,
                                                     index_dir=index_path_ssd)
ssd_time = t.tac()
print "ssd time: {}".format(ssd_time)

print 'Spin clustering...'
t.reset()
t.tic()
nclusts_spin, labels_spin = myEAC.sp_sl_lifetime_disk(coassc_path_spin,
                                                     n_partitions,
                                                     n_clusters=0,
                                                     index_dir=index_path_spin)
spin_time = t.tac()
print "spin time: {}".format(spin_time)
