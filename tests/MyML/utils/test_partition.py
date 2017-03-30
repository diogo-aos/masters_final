import numpy as np
from MyML.utils.partition import *


def test_convertIndexToBin_small():
    # prepare data
    part_bin = [[0,1,0,1], [1,0,1,0]]
    part_lst = [[1,3], [0,2]]

    part_bin = np.array(part_bin, dtype=np.uint8)
    part_lst = [np.array(a, dtype=np.uint64) for a in part_lst]

    # call function
    out = convertIndexToBin(part_lst)

    assert np.all(part_bin == out)


def test_convertIndexToBin():
    # prepare data
    clust_size, n_samples = 200, 100000
    part_lst = []
    part_bin = []
    available_idx = [x for x in range(n_samples)]
    while available_idx:
        clust_n_samples = np.random.randint(1, 200)
        clust = []
        for i in range(clust_n_samples):
            if not available_idx:
                break
            x = available_idx.pop(np.random.randint(0, len(available_idx)))
            clust.append(x)
        clust = np.array(clust, dtype=np.uint64)
        part_lst.append(clust)

        clust_bin = np.zeros(n_samples, dtype=np.uint8)
        clust_bin[clust] = 1
        part_bin.append(clust_bin)

    part_bin = np.array(part_bin, dtype=np.uint8)

    # call function
    out = convertIndexToBin(part_lst)

    assert np.all(part_bin == out)

def test_convertClusterStringToBin_small_min0():
    # prepare data
    part_bin = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cluster id = 0
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # cluster id = 1
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # cluster id = 2
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],  # cluster id = 3
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]]  # cluster id = 4
    part_lst = [0, 1, 1, 3, 2, 4, 2, 4, 3, 3]

    part_bin = np.array(part_bin, dtype=np.uint8)
    part_lst = np.array(part_lst, dtype=np.uint64)

    # call function
    out = convertClusterStringToBin(part_lst)

    assert np.all(part_bin == out)

def test_convertClusterStringToBin_small_min1():
    # prepare data
    part_bin = [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # cluster id = 1
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # cluster id = 2
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],  # cluster id = 3
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]]  # cluster id = 4
    part_lst =  [1, 1, 1, 3, 2, 4, 2, 4, 3, 3]

    part_bin = np.array(part_bin, dtype=np.uint8)
    part_lst = np.array(part_lst, dtype=np.uint64)

    # call function
    out = convertClusterStringToBin(part_lst)

    assert np.all(part_bin == out)


def test_convertClusterStringToIndex_small():
    # prepare data
    part_lst =  [1, 1, 1, 3, 2, 4, 2, 4, 3, 3]
    part_idx = [
                [0, 1, 2],  # cluster id = 1
                [4, 6],     # cluster id = 2
                [3, 8, 9],  # cluster id = 3
                [5, 7]      # cluster id = 4
                ]

    part_idx = [np.array(a, dtype=np.uint64) for a in part_idx]
    part_lst = np.array(part_lst, dtype=np.uint32)

    # call function
    out = convertClusterStringToIndex(part_lst)

    assert np.all([np.all(a == b) for a, b in zip(part_idx, out)])
