import unittest

import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from numba import cuda, njit, jit, int32, float32

from MyML.utils.profiling import Timer

from MyML.graph.mst import boruvka_minho_seq,\
                     boruvka_minho_gpu,\
                     compute_cuda_grid_dim
from MyML.graph.connected_components import connected_comps_seq,\
                                      connected_comps_gpu
from MyML.graph.build import getGraphFromEdges_gpu, getGraphFromEdges_seq


import os.path

module_path = os.path.dirname(__file__)

path_4elt = os.path.join(module_path, 'datasets/graphs/4elt.npz')
path_usa_cal = os.path.join(module_path, 'datasets/graphs/USA-CAL.npz')


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


@njit
def outdegree_from_firstedge(firstedge, outdegree, n_edges):
    n_vertices = firstedge.size
    for v in range(n_vertices - 1):
        outdegree[v] = firstedge[v + 1] - firstedge[v]
    outdegree[n_vertices - 1] = n_edges - firstedge[n_vertices - 1]


def load_graph_for_boruvka(filename):
    sp_mat = load_sparse_csr(filename)
    dest = sp_mat.indices
    weight = sp_mat.data
    firstedge = sp_mat.indptr[:-1]
    outdegree = np.empty_like(firstedge)
    outdegree_from_firstedge(firstedge, outdegree, dest.size)
    return dest, weight, firstedge, outdegree


class TestMethods(unittest.TestCase):
    # def setUp(self):
    #     cuda.current_context().trashing.clear()

    def tearDown(self):
        cuda.current_context().trashing.clear()

    def test_boruvka_sequential_4elt(self):

        # get MST from Boruvka algorithm
        graph = load_graph_for_boruvka(path_4elt)
        dest, weight, firstedge, outdegree = graph

        n_edges = dest.size
        n_vertices = firstedge.size

        t1 = Timer()
        t1.tic()
        mst, n_mst = boruvka_minho_seq(dest, weight, firstedge, outdegree)
        t1.tac()

        if n_mst < mst.size:
            mst = mst[:n_mst]

        # print 'graph nodes: ', n_edges
        # print 'MST edges: ', n_mst

        # get MST from scipy library
        graph_csr = load_sparse_csr(path_4elt)
        scipy_mst = minimum_spanning_tree(graph_csr)
        true_mst_size = scipy_mst.size

        assert_msg = 'MST number of edges mismatch'
        self.assertEqual(n_mst, true_mst_size, assert_msg)

        assert_msg = 'MST total weight mismatch'
        self.assertEqual(weight[mst].sum(), scipy_mst.sum(), assert_msg)


        # print "time elapsed: ", t1.elapsed
        # mst has to be sorted for comparison with device mst because different
        # threads might be able to write first
        # mst.sort()
        # print mst
        # print n_edges

    def test_boruvka_sequential_usa_cal(self):
        # get MST from Boruvka algorithm
        graph = load_graph_for_boruvka(path_usa_cal)
        dest, weight, firstedge, outdegree = graph

        n_edges = dest.size
        n_vertices = firstedge.size

        t1 = Timer()
        t1.tic()
        mst, n_mst = boruvka_minho_seq(dest, weight, firstedge, outdegree)
        t1.tac()

        if n_mst < mst.size:
            mst = mst[:n_mst]

        # get MST from scipy library
        graph_csr = load_sparse_csr(path_usa_cal)
        scipy_mst = minimum_spanning_tree(graph_csr)
        true_mst_size = scipy_mst.size

        assert_msg = 'MST number of edges mismatch'
        self.assertEqual(n_mst, true_mst_size, assert_msg)

        assert_msg = 'MST total weight mismatch'
        self.assertEqual(weight[mst].sum(), scipy_mst.sum(), assert_msg)

    def test_device_boruvka_4elt(self):

        sp_mat = load_sparse_csr(path_4elt)
        dest = sp_mat.indices
        weight = sp_mat.data
        firstedge = sp_mat.indptr[:-1]
        outdegree = np.empty_like(firstedge)
        outdegree_from_firstedge(firstedge, outdegree, dest.size)

        n_edges = dest.size
        n_vertices = firstedge.size

        t1 = Timer()
        t1.tic()
        mst, n_mst = boruvka_minho_gpu(dest, weight, firstedge, outdegree)
        t1.tac()

        if n_mst < mst.size:
            mst = mst[:n_mst]

        # get MST from scipy library
        graph_csr = load_sparse_csr(path_4elt)
        scipy_mst = minimum_spanning_tree(graph_csr)
        true_mst_size = scipy_mst.size

        assert_msg = 'MST number of edges mismatch'
        self.assertEqual(n_mst, true_mst_size, assert_msg)

        assert_msg = 'MST total weight mismatch'
        self.assertEqual(weight[mst].sum(), scipy_mst.sum(), assert_msg)

        # print "time elapsed: ", t1.elapsed
        # mst.sort()
        # print mst
        # print n_edges

    def test_device_boruvka_usa_cal(self):

        sp_mat = load_sparse_csr(path_usa_cal)
        dest = sp_mat.indices
        weight = sp_mat.data
        firstedge = sp_mat.indptr[:-1]
        outdegree = np.empty_like(firstedge)
        outdegree_from_firstedge(firstedge, outdegree, dest.size)

        n_edges = dest.size
        n_vertices = firstedge.size

        t1 = Timer()
        t1.tic()
        mst, n_mst = boruvka_minho_gpu(dest, weight, firstedge, outdegree)
        t1.tac()

        if n_mst < mst.size:
            mst = mst[:n_mst]

        # get MST from scipy library
        graph_csr = load_sparse_csr(path_usa_cal)
        scipy_mst = minimum_spanning_tree(graph_csr)
        true_mst_size = scipy_mst.size

        assert_msg = 'MST number of edges mismatch'
        self.assertEqual(n_mst, true_mst_size, assert_msg)

        assert_msg = 'MST total weight mismatch'
        self.assertEqual(weight[mst].sum(), scipy_mst.sum(), assert_msg)

        # print "time elapsed: ", t1.elapsed
        # mst.sort()
        # print mst
        # print n_edges

    def test_seq_gpu(self):
        print "HOST VS DEVICE"

        same_sol = list()
        same_cost = list()

        for r in range(20):
            sp_mat = load_sparse_csr(path_4elt)
            dest = sp_mat.indices
            weight = sp_mat.data
            firstedge = sp_mat.indptr[:-1]  # last element is the total number
            outdegree = np.empty_like(firstedge)
            outdegree_from_firstedge(firstedge, outdegree, dest.size)

            n_edges = dest.size
            n_vertices = firstedge.size

            t1, t2 = Timer(), Timer()

            t1.tic()
            mst1, n_edges1 = boruvka_minho_seq(dest, weight,
                                               firstedge, outdegree)
            t1.tac()

            if n_edges1 < mst1.size:
                mst1 = mst1[:n_edges1]
            mst1.sort()

            assert_msg = '4elt dataset MST not fully connected in sequential'
            self.assertEqual(mst1.size, n_vertices-1, assert_msg)

            t2.tic()
            mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstedge,
                                               outdegree, MAX_TPB=256)
            t2.tac()

            if n_edges2 < mst2.size:
                mst2 = mst2[:n_edges2]
            mst2.sort()

            assert_msg = '4elt dataset MST not fully connected in gpu'
            self.assertEqual(mst2.size, n_vertices-1, assert_msg)

            # how many edges are common to both solutions
            # same_sol.append(np.in1d(mst1, mst2).sum())

            # check MST cost
            cost1 = weight[mst1].sum()
            cost2 = weight[mst2].sum()
            self.assertEqual(cost1, cost2, 'MSTs have diferent costs')
            # same_cost.append(cost1 == cost2)


if __name__ == '__main__':
    unittest.main()
