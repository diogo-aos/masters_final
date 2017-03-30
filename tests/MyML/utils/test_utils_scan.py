# -*- coding: utf-8 -*-

"""
This is a test for the last block scan.
"""


import MyML.utils.scan as MyScan
import numpy as np
from numba import cuda
from timeit import default_timer as timer


ARRAY_SIZE = 100000


class TestScanCPU():

    def test_exprefixsumNumba_init_0(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        out_ary = np.empty_like(in_ary)

        init = 0

        output = MyScan.exprefixsumNumba(in_ary, out_ary, init=init)

        # check last carry
        assert np.isclose(output, in_ary.sum()), 'carry return is not sum'

        carry = init
        for i in xrange(ARRAY_SIZE):
            assert out_ary[i] == carry, 'output array not correct'
            carry += in_ary[i]

    def test_exprefixsumNumba_init_random(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        out_ary = np.empty_like(in_ary)

        init = np.random.randint(100000)

        output = MyScan.exprefixsumNumba(in_ary, out_ary, init=init)

        # check last carry
        assert np.isclose(output - init, in_ary.sum()), 'carry return is not sum'

        carry = init
        for i in xrange(ARRAY_SIZE):
            assert out_ary[i] == carry, 'output array not correct'
            carry += in_ary[i]

    def test_exprefixsumNumbaSingle_init_0(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        in_ary_cpy = in_ary.copy()

        init = 0

        output = MyScan.exprefixsumNumbaSingle(in_ary, init=init)

        # check last carry
        assert np.isclose(output, in_ary_cpy.sum()), 'carry return is not sum'

        # check array
        carry = init
        for i in xrange(ARRAY_SIZE):
            assert in_ary[i] == carry, 'output array not correct'
            carry += in_ary_cpy[i]

    def test_exprefixsumNumbaSingle_init_random(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        in_ary_cpy = in_ary.copy()

        init = np.random.randint(100000)

        output = MyScan.exprefixsumNumbaSingle(in_ary, init=init)

        # check last carry
        assert np.isclose(output - init, in_ary_cpy.sum()), 'carry return is not sum'

        # check array
        carry = init
        for i in xrange(ARRAY_SIZE):
            assert in_ary[i] == carry, 'output array not correct'
            carry += in_ary_cpy[i]


class TestScanGPU():

    def test_scan_int32(self):
        in_ary = np.random.randint(0, 10000, ARRAY_SIZE).astype(np.int32)
        in_ary_d = cuda.to_device(in_ary)

        output = MyScan.scan_gpu(in_ary_d)

        out_carry = output.getitem(0)
        out_ary = in_ary_d.copy_to_host()

        # check last carry
        assert np.isclose(out_carry, in_ary.sum()), 'carry return is not sum'

        # check array
        carry = 0
        for i in xrange(ARRAY_SIZE):
            assert out_ary[i] == carry, 'output array not correct'
            carry += in_ary[i]

    def test_scan_int64(self):
        in_ary = np.random.randint(0, 10000, ARRAY_SIZE)
        in_ary_d = cuda.to_device(in_ary)

        output = MyScan.scan_gpu(in_ary_d)

        out_carry = output.getitem(0)
        out_ary = in_ary_d.copy_to_host()

        # check last carry
        assert np.isclose(out_carry, in_ary.sum()), 'carry return is not sum'

        # check array
        carry = 0
        for i in xrange(ARRAY_SIZE):
            assert out_ary[i] == carry, 'output array not correct'
            carry += in_ary[i]

    def test_scan_fp32(self):
        in_ary = np.random.rand(ARRAY_SIZE).astype(np.float32)
        in_ary_d = cuda.to_device(in_ary)

        output = MyScan.scan_gpu(in_ary_d)

        out_carry = output.getitem(0)
        out_ary = in_ary_d.copy_to_host()

        # check last carry
        print out_carry, in_ary.sum()
        # self.assertTrue(np.isclose(out_carry, in_ary.sum()),
        #                 'carry return is not sum')

        # check array
        carry = 0
        for i in xrange(ARRAY_SIZE):
            assert out_ary[i] == carry, 'output array not correct'
            carry += in_ary[i]

    def test_scan_fp64(self):
        in_ary = np.random.rand(ARRAY_SIZE).astype(np.float64)
        in_ary_d = cuda.to_device(in_ary)

        output = MyScan.scan_gpu(in_ary_d)

        out_carry = output.getitem(0)
        out_ary = in_ary_d.copy_to_host()

        # check last carry
        print out_carry, in_ary.sum()
        # self.assertTrue(np.isclose(out_carry, in_ary.sum()),
        #                 'carry return is not sum')

        # check array
        carry = 0
        for i in xrange(ARRAY_SIZE):
            self.assertEqual(out_ary[i], carry, 'output array not correct')
            carry += in_ary[i]


def test_last_block():

    MAX_TPB = 512
    n = 1024

    a = np.arange(n).astype(np.int32)
    reference = np.empty_like(a)

    start = timer()
    MyScan.exprefixsumNumba(a, reference, init=0)
    end = timer()

    auxidx = -1

    elb = a.size
    p2elb = np.int(np.ceil(np.log2(elb)))
    telb = 2 ** p2elb
    tlb = telb / 2
    startIdx = 0

    sm_size = telb * a.itemsize

    aux = np.empty(1, dtype=np.int8)

    trash = cuda.device_array(1)

    e1, e2 = cuda.event(), cuda.event()

    e1.record()
    MyScan.last_scan[1, tlb, 0, sm_size](a, aux, -1, elb, startIdx)
    e2.record()

    print "CPU took:    ", (end - start) * 1000, " ms"
    print "Kernel took: ", cuda.event_elapsed_time(e1, e2), " ms"

    print (a == reference).all()


def test_recursive_big_scan():

    print "running recursive scan test"

    MAX_TPB = 512
    n = 2e6
    n = int(n)

    a = np.arange(n).astype(np.int32)
    reference = np.empty_like(a)

    start = timer()
    sum_ref = MyScan.exprefixsumNumba(a, reference, init=0)
    end = timer()

    dA = cuda.to_device(a)

    # e1, e2 = cuda.event(), cuda.event()

    # e1.record()
    # e2.record()

    start2 = timer()
    total_sum = MyScan.scan_gpu(dA)
    end2 = timer()

    dA.copy_to_host(ary=a)
    sum_gpu = total_sum.copy_to_host()

    print "sum_ref = ", sum_ref
    print "sum_gpu = ", sum_gpu

    print "CPU took:    ", (end - start) * 1000, " ms"
    print "Kernel took: ", (end2 - start2) * 1000, " ms"

    print (a == reference).all()


def test_recursive_step_by_step():

    ## setup

    MAX_TPB = 512
    n = 5000

    a = np.arange(n).astype(np.int32)
    reference = np.empty_like(a)

    start = timer()
    sum_ref = MyScan.exprefixsumNumba(a, reference, init=0)
    end = timer()

    dA = cuda.to_device(a)

    # e1, e2 = cuda.event(), cuda.event()
    # e1.record()
    # e2.record()


    ## scan
    in_ary = dA

    epb = MAX_TPB * 2
    whole_blocks = n // epb
    el_last_block = n % epb

    n_scans = whole_blocks
    if el_last_block != 0:
        n_scans += 1

    ## prescan

    dAux = cuda.device_array(shape=n_scans, dtype=np.int32)
    sm_size = epb * in_ary.dtype.itemsize

    MyScan.prescan[whole_blocks, MAX_TPB, 0, sm_size](in_ary, dAux)

    # tIn = in_ary.copy_to_host()
    # tAux = dAux.copy_to_host()

    p2elb = np.int(np.ceil(np.log2(el_last_block)))
    p2_el_last_block = 2 ** p2elb  # the smallest number of elements that is power of 2
    tlb = p2_el_last_block >> 1  # number of threads in last block

    sm_size = p2_el_last_block * in_ary.dtype.itemsize

    startIdx = n - el_last_block
    auxIdx = n_scans - 1

    MyScan.last_scan[1, tlb, 0, sm_size](in_ary, dAux, auxIdx,
                                       el_last_block, startIdx)

    in_ary2 = dAux
    n2 = in_ary2.size

    if n2 < MAX_TPB << 1:
        el_last_block2 = n2

        p2elb2 = np.int(np.ceil(np.log2(el_last_block2)))
        p2_el_last_block2 = 2 ** p2elb  # the smallest number of elements that is power of 2
        tlb2 = p2_el_last_block2 >> 1  # number of threads in last block

        total_sum = cuda.device_array(shape=1, dtype=np.int32)
        sm_size2 = p2_el_last_block2 * in_ary2.dtype.itemsize

        startIdx2 = 0
        auxIdx2 = 0

        MyScan.last_scan[1, tlb2, 0, sm_size2](in_ary2, total_sum, auxIdx2, el_last_block2, startIdx2)

    MyScan.scan_sum[n_scans, tlb](in_ary, dAux)

    tIn = in_ary.copy_to_host()
    tAux = dAux.copy_to_host()
    tSum = total_sum.copy_to_host()

    print "finish"


def test_prescan():

    a = np.arange(2048).astype(np.int32)
    reference = np.empty_like(a)

    ref_sum = MyScan.exprefixsumNumba(a, reference)

    a1 = np.arange(1024).astype(np.int32)
    a2 = np.arange(1024, 2048).astype(np.int32)

    ref1 = np.empty_like(a1)
    ref2 = np.empty_like(a2)

    ref_sum1 = MyScan.exprefixsumNumba(a1, ref1)
    ref_sum2 = MyScan.exprefixsumNumba(a2, ref2)

    dAux = cuda.device_array(2, dtype=np.int32)
    dA = cuda.to_device(a)

    sm_size = 1024 * a.dtype.itemsize

    MyScan.prescan[2, 512, 0, sm_size](dA, dAux)

    aux = dAux.copy_to_host()
    a_gpu = dA.copy_to_host()

    print "finish"
