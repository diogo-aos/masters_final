#
# nohup python experiments/CUDA/benchmark_K_Means_2.py -f experiments/CUDA/ -t 0.4 -tpb 512 -hm 6000 -y >&/dev/null &
#


# coding: utf-8
import argparse
import os.path
import sys
import numpy as np
from numba import cuda

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   PARSE ARGUMENTS AND CHECK FOLDER VALIDITY
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="where to save results to", type=str)

parser.add_argument('-hm', '--hostmemory', help='host memory that program can '
                    'use in GB', type=float, required=True)

parser.add_argument("-tpb", "--threadsperblock", help="maximum threads per block"
                    " for GPU", type=int, default=0)

parser.add_argument("-t", "--thresholdgpu", help="threshold for gpu mem usage",
                    type=float, default=0.97)

parser.add_argument('-ppt', '--pointsperthread', help='points to process per '
                    'perthread', type=int, default=1)

parser.add_argument("-tn", "--testname", help="name of test, prefixed to files",
                    type=str, default='')

parser.add_argument("-rt", "--runtime", help="how many hours to run for",
                    type=float, default=0)



parser.add_argument("-r", "--rounds", help="number of rounds",
                    type=int, default=5)

parser.add_argument('-y', "--yes", help="don't ask confirmation of folder",
                    action='store_true')

parser.add_argument("-c", "--cardinalitystart", help="start at this cardinality, non inclusive",
                    type=int, default=0)

parser.add_argument("-ce", "--cardinalityend", help="end at this cardinality, non inclusive",
                    type=int, default=2**32)

parser.add_argument('-rb', "--reboot", action='store_true',
                    help="reboot from previous state. Must specify previous"\
                         " state with -rp arguments")

parser.add_argument('-rp', "--reboot_parameters", type=str,
                    default=False, nargs=6,
                    help='reboot arguments: [#samples] [rule name] '\
                         '[type of mat] [round] [results index]')

args = parser.parse_args()
test_name = args.testname
folder = args.folder
hostmemory = args.hostmemory
threadsperblock = args.threadsperblock
thresholdgpu = args.thresholdgpu
runtime = args.runtime
PPT = args.pointsperthread
cardinality_start = args.cardinalitystart
cardinality_end = args.cardinalityend
arg_rounds = args.rounds

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

# await confirmation
if not args.yes:
    raw_input("Folder: {}\nIs this correct?".format(folder))
else:
    print "Folder being used is: {}".format(folder)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   IMPORTS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import MyML.utils.profiling as myProf
t = myProf.Timer()
t.tic()

import numpy as np
from numba import cuda
import pandas as pd # for storing results

from sklearn import datasets # generate gaussian mixture

import MyML.cluster.K_Means3 as myKM
myKM.CUDA_PPT = PPT

# Setup logging
import logging

# util modules
import gc

print "Import time:", t.tac()


stop_runtime = True if runtime > 0 else False
if stop_runtime:
    runtime_timer = myProf.Timer()
    runtime_total = runtime * 3600 # runtime in seconds
    runtime_timer.tic()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   SET UP LOGGING
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Status logging
logger = logging.getLogger('status')
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file handler
handler = logging.FileHandler(os.path.join(folder,'kmeans_{}.log'.format(test_name)))
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

# create a console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(consoleHandler)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   SET UP EXPERIMENT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

results_filename = os.path.join(folder,'kmeans_results_{}.csv'.format(test_name))


def generateData(n,d):
    "Data set by numpy.arange"
    return np.arange(n*d, dtype=np.float32)

def generate_distribution_data(n, d):
    """Random data from uniform distribution in the interval [0.0,1.0[."""
    data = np.empty((n,d), dtype=np.float32)
    f = n / 10
    for i in range(1, 10):
        start = (i - 1) * f
        end = i * f
        data[start:end] = np.random.random((f, d))
    data[end:] = np.random.random((n-end, d))

    return data

def logged_generateData(n,d, fn):
    logger.info("Generating dataset...")
    logger.info(fn.__doc__)
    logger.info("Cardinality: {}".format(n))
    logger.info("Dimensionality: {}".format(d))
    return fn(n,d)

# computes required device memory: data + labels + dists + centroids
# n = cardinality, d = dimensionality, c = number of clusters
req_mem = lambda n, d, c: (n*d*4 + n*2*4 + c*d*4)

# HOST memory max
MAX_ALLOWED_HOST_MEM = hostmemory * 2**30 
MAX_ALLOWED_HOST_MEM = int(MAX_ALLOWED_HOST_MEM)

# compute device memory
c = cuda.current_context()
free_mem, total_mem = c.get_memory_info()
MAX_ALLOWED_DEVICE_MEM = thresholdgpu * total_mem # threshold default is 0.97
MAX_ALLOWED_DEVICE_MEM = int(MAX_ALLOWED_DEVICE_MEM)

logger.info("Will occupy maximum of {} MB in"
            " device memory.".format(MAX_ALLOWED_DEVICE_MEM / (1024.0**2)))

# cardinality = [100, 250, 500, 750,
#                1e3, 2.5e3, 5e3, 7.5e3,
#                1e4, 2.5e4, 5e4, 7.5e4,
#                1e5, 2.5e5, 5e5, 7.5e5,
#                1e6, 2.5e6, 5e6, 7.5e6,
#                1e7]
cardinality = [100, 1e3, 5e3, 1e4,
               2.5e4, 5e4, 7.5e4, 1e5,
               2.5e5, 5e5, 7.5e5, 1e6,
               2.5e6, 5e6, 1e7]
cardinality = map(int,cardinality)

# dimensionality = [2, 5,
#                   10, 25, 50, 75,
#                   100, 250, 500, 750,
#                   1000, 1500]

# dimensionality = [2, 20,
#                   200, 1000]         

dimensionality = [2, 200]

# clusters = [2, 4, 6, 8, 10,
#             20, 40, 60, 80, 100,
#             150, 200, 250, 300, 350, 400, 450, 500,
#             600, 700, 800, 900, 1000]

clusters = [2, 4, 8, 16, 32, 64, 256, 512, 1024, 1536, 2048]

n_iters = [3]
rounds = arg_rounds

label_centroid_pairs = {'python': ('python','python'),
                        'numpy': ('numpy','numpy'),
                        'numba': ('numba','numba'),
                        'cuda': ('cuda','numba')}

del label_centroid_pairs['python'] #don't compute with python
del label_centroid_pairs['numpy'] #don't compute with numpy   
#del label_centroid_pairs['numba'] #don't compute with CPU Numba   

# cardinality - number of samples
# dimensionality - number of dimensions
# number of iterations (of the K-Means algorithm)
# label mode - python, numpy, numba or CUDA for computing labels
# centroid mode - python, numpy or numba for recomputing the centroids
# total time that took the algorithm to run
# time per iteration = total time / number of iterations

res_cols = ['cardinality', 'dimensionality', 'number of clusters',
            'number of iterations', 'label mode', 'centroid mode',
            'round', 'total time', 'label time', 'centroid time',
            'cum_data_transfer', 'n_data_transfer',
            'std_data_transfer', 'max_data_transfer', 'min_data_transfer',
            'cum_centroids_transfer', 'n_centroids_transfer',
            'std_centroids_transfer', 'max_centroids_transfer', 'min_centroids_transfer',
            'cum_labels_transfer', 'n_labels_transfer',
            'std_labels_transfer', 'max_labels_transfer', 'min_labels_transfer',
            'cum_dists_transfer', 'n_dists_transfer',
            'std_dists_transfer', 'max_dists_transfer', 'min_dists_transfer',
            'cum_kernel', 'n_kernel',
            'std_kernel', 'max_kernel', 'min_kernel']

res_lines = (len(cardinality) * len(dimensionality) * len(clusters)
                * len(n_iters) * len(label_centroid_pairs) * rounds)

results = pd.DataFrame(index=range(res_lines), columns=res_cols)

# create timers
timer_list = [myProf.Timer() for i in range(3)]
t_total, t_label, t_centroids = timer_list

# state variables

# if a certain mode takes more than this ammount of seconds, then it will not
# be executed for bigger datasets 
mode_stop_times = {'python': 1800,
                   'numpy': 1800,
                   'numba':1800,
                   'cuda':1800}

# for saving the sizes after which
mode_stop_sizes = {'python': 0,
                   'numpy': 0,
                   'numba': 0,
                   'cuda': 0
                   }


round_time_change = 60 # after this many seconds, change to 1 round
round_size_change = {'python': np.inf,
                     'numpy': np.inf,
                     'numba': np.inf,
                     'cuda': np.inf}

# function used to convert ms to sec
to_sec = lambda x:x/1000.0

if reboot:
    n_old = int(reboot_parameters[0])
    d_old = int(reboot_parameters[1])
    nc_old = int(reboot_parameters[2])
    lm_old = reboot_parameters[3]
    r_old = int(reboot_parameters[4])
    idx_old = int(reboot_parameters[5])

    if lm_old not in mode_stop_times.keys():
        raise ValueError("Label mode not exist.")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   make sure all code is compiled for "real" runs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
data = generate_distribution_data(1000, 2)

generator = myKM.K_Means(label_mode="numba", centroid_mode="numba",
                         max_iter=3)
generator.fit(data)

generator = myKM.K_Means(label_mode="cuda", centroid_mode="numba",
                         max_iter=3)
generator._PPT = PPT
if not threadsperblock:
    generator._MAX_THREADS_BLOCK = threadsperblock

generator.fit(data)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   START EXPERIMENT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
logger.info("Starting experiment...")
if reboot:
    results = pd.read_csv(os.path.join(folder, 'kmeans_results.csv'))
    res_idx = idx_old + 1
else:
    res_idx = 0
for n in cardinality: # for each size of dataset

    if n < cardinality_start:
        continue

    if n > cardinality_end:
        continue

    if reboot and n != n_old:
        continue

    for d in dimensionality: # for each dimension
        if reboot and d != d_old:
            continue

        # sample data
        # data_sampled = np.ascontiguousarray(data[::d])
        data_sampled = logged_generateData(n, d, generate_distribution_data)
        n = data_sampled.shape[0]

        logger.info("Dataset of {} patterns and {} dimensions.".format(n,d))

        for nc in clusters: # for each number of clusters

            if reboot and nc != nc_old:
                continue

            # if too many clusters for number of samples, don't run
            if nc >= int(0.75 * n):
                continue

            memory_used = req_mem(n, d, nc)

            if memory_used > MAX_ALLOWED_DEVICE_MEM:
                logger.info("Wouldn't fit in device. Skipping...")
                continue        

            for it in n_iters: # for each number of iterations

                # for each pair of computing modes (label, centroid modes)
                for key, lcm in label_centroid_pairs.iteritems():

                    lm, cm = lcm # lm = label mode, cm = centroid mode

                    if reboot and lm != lm_old:
                        continue

                    # separator for different configurations
                    logger.info("- - - - - - - - - - - - - - - - -")
                    logger.info("- - - - - - - - - - - - - - - - -")                    

                    # checks if mode should has been set to be skipped for big
                    # times; checks if dataset memory is as bis as the last time
                    # it made this mode exceed maximum times allowed
                    if (mode_stop_times.has_key(key)
                            and mode_stop_sizes.get(key) >= memory_used):
                        continue

                    if memory_used >= round_size_change[key]:
                        crounds = 1
                    else:
                        crounds = rounds

                    for r in range(crounds): # for each round

                        if reboot and r_old > crounds:
                            r_old = crounds - 1

                        if reboot and r != r_old:
                            continue
                        else:
                            reboot = False

                        # new generator
                        generator = myKM.K_Means(label_mode=lm, centroid_mode=cm,
                                                 max_iter=it, n_clusters=nc)
                        generator._PPT = PPT
                        if not threadsperblock:
                            generator._MAX_THREADS_BLOCK = threadsperblock                    

                        # round separator
                        logger.info("- - - - - - - - - - - - - - - - -")

                        logger.info("Cardinality:{}".format(n))
                        logger.info("Dimensionality:{}".format(d))
                        logger.info("Number of clusters:{}".format(nc))
                        logger.info("Number of iterations:{}".format(it))
                        logger.info("Label mode:{}".format(lm)) 
                        logger.info("Centroid mode:{}".format(cm))
                        logger.info("Round:{}".format(r))

                        results.loc[res_idx,'cardinality'] = n
                        results.loc[res_idx,'dimensionality'] = d
                        results.loc[res_idx,'number of clusters'] = nc
                        results.loc[res_idx,'number of iterations'] = it
                        results.loc[res_idx,'label mode'] = lm
                        results.loc[res_idx,'centroid mode'] = cm
                        results.loc[res_idx,'round'] = r

                        # set up timers
                        map(myProf.Timer.reset, timer_list) # reset all timers
                        generator._label = t_label.wrap_function(generator._label)
                        generator._recompute_centroids = t_centroids.wrap_function(generator._recompute_centroids)
                        generator.fit = t_total.wrap_function(generator.fit)

                        generator.fit(data_sampled) # perform K-Means

                        results.loc[res_idx, 'total time'] = t_total.elapsed
                        results.loc[res_idx, 'label time'] = t_label.elapsed
                        results.loc[res_idx, 'centroid time'] = t_centroids.elapsed
                        logger.info("Total time:{}".format(t_total.elapsed))

                        if lm == 'cuda':
                            data_transfer_times = map(to_sec, generator.man_prof['data_timings']) # convert to sec
                            results.loc[res_idx, 'cum_data_transfer'] = sum(data_transfer_times)
                            results.loc[res_idx, 'std_data_transfer'] = np.std(data_transfer_times)
                            results.loc[res_idx, 'max_data_transfer'] = np.max(data_transfer_times)
                            results.loc[res_idx, 'min_data_transfer'] = np.min(data_transfer_times)
                            results.loc[res_idx, 'n_data_transfer'] = len(data_transfer_times)

                            centroids_transfer_times = map(to_sec, generator.man_prof['centroids_timings']) # convert to sec
                            results.loc[res_idx, 'cum_centroids_transfer'] = sum(centroids_transfer_times)
                            results.loc[res_idx, 'std_centroids_transfer'] = np.std(centroids_transfer_times)
                            results.loc[res_idx, 'max_centroids_transfer'] = np.max(centroids_transfer_times)
                            results.loc[res_idx, 'min_centroids_transfer'] = np.min(centroids_transfer_times)
                            results.loc[res_idx, 'n_centroids_transfer'] = len(centroids_transfer_times)

                            kernel_times = map(to_sec, generator.man_prof['kernel_timings']) # convert to sec
                            results.loc[res_idx, 'cum_kernel'] = sum(kernel_times)
                            results.loc[res_idx, 'std_kernel'] = np.std(kernel_times)
                            results.loc[res_idx, 'max_kernel'] = np.max(kernel_times)
                            results.loc[res_idx, 'min_kernel'] = np.min(kernel_times)
                            results.loc[res_idx, 'n_kernel'] = len(kernel_times)

                            labels_transfer_times = map(to_sec, generator.man_prof['labels_timings']) # convert to sec
                            results.loc[res_idx, 'cum_labels_transfer'] = sum(labels_transfer_times)
                            results.loc[res_idx, 'std_labels_transfer'] = np.std(labels_transfer_times)
                            results.loc[res_idx, 'max_labels_transfer'] = np.max(labels_transfer_times)
                            results.loc[res_idx, 'min_labels_transfer'] = np.min(labels_transfer_times)
                            results.loc[res_idx, 'n_labels_transfer'] = len(labels_transfer_times)

                            dists_transfer_times = map(to_sec, generator.man_prof['dists_timings']) # convert to sec
                            results.loc[res_idx, 'cum_dists_transfer'] = sum(dists_transfer_times)
                            results.loc[res_idx, 'std_dists_transfer'] = np.std(dists_transfer_times)
                            results.loc[res_idx, 'max_dists_transfer'] = np.max(dists_transfer_times)
                            results.loc[res_idx, 'min_dists_transfer'] = np.min(dists_transfer_times)
                            results.loc[res_idx, 'n_dists_transfer'] = len(generator.man_prof['dists_timings'])

                        results.to_csv(results_filename)
                        res_idx += 1

                        # free resources of current context
                        # cuda.current_context().reset()
                        # cuda.close()
                        # for val in [key for key in cuda.current_context().allocations.values()]:
                        #     cuda.current_context().memfree(val.device_pointer)
                        for cpounter_key in [cpounter_key for cpounter_key in cuda.current_context().allocations.keys()]:
                            del cuda.current_context().allocations[cpounter_key]
                        cuda.current_context().trashing.clear()

                        # update memory for round change
                        if (crounds == rounds and 
                            key in round_size_change.keys() and
                            t_total.elapsed >= round_time_change):
                            round_size_change[key] = memory_used

                        # if time exceeded limit in defined mode, save size of
                        # the dataset; -1 if default time for undefined modes
                        if t_total.elapsed >= mode_stop_times.get(key, -1):
                            mode_stop_sizes[key] = memory_used
                            break
                            
                        if stop_runtime:
                            runtime_timer.tac()
                            runtime_timer.tic()
                            if runtime_timer.elapsed >= runtime_total:
                                t_elapsed = runtime_timer.elapsed
                                logger.info('Time\'s up! Total time: {}\n'\
                                            'Last results index '\
                                            'was {}'.format(t_elapsed, res_idx))
                                sys.exit(0)

                        del generator
        del data_sampled