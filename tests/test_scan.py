# -*- coding: utf-8 -*-

"""
This is a test for the last block scan.
"""


import MyML.utils.scan as scan
import numpy as np
from numba import cuda
from timeit import default_timer as timer

import unittest


ARRAY_SIZE = 100000


class TestScanCPU(unittest.TestCase):

    def test_exprefixsumNumba_init_0(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        out_ary = np.empty_like(in_ary)

        init = 0

        output = scan.exprefixsumNumba(in_ary, out_ary, init=init)

        # check last carry
        self.assertTrue(np.isclose(output, in_ary.sum()),
                        'carry return is not sum')

        carry = init
        for i in xrange(ARRAY_SIZE):
            self.assertEqual(out_ary[i], carry, 'output array not correct')
            carry += in_ary[i]

    def test_exprefixsumNumba_init_random(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        out_ary = np.empty_like(in_ary)

        init = np.random.randint(100000)

        output = scan.exprefixsumNumba(in_ary, out_ary, init=init)

        # check last carry
        self.assertTrue(np.isclose(output - init, in_ary.sum()),
                        'carry return is not sum')

        carry = init
        for i in xrange(ARRAY_SIZE):
            self.assertEqual(out_ary[i], carry, 'output array not correct')
            carry += in_ary[i]

    def test_exprefixsumNumbaSingle_init_0(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        in_ary_cpy = in_ary.copy()

        init = 0

        output = scan.exprefixsumNumbaSingle(in_ary, init=init)

        # check last carry
        self.assertTrue(np.isclose(output, in_ary_cpy.sum()),
                        'carry return is not sum')

        # check array
        carry = init
        for i in xrange(ARRAY_SIZE):
            self.assertEqual(in_ary[i], carry, 'output array not correct')
            carry += in_ary_cpy[i]

    def test_exprefixsumNumbaSingle_init_random(self):
        in_ary = np.random.rand(ARRAY_SIZE)
        in_ary_cpy = in_ary.copy()

        init = np.random.randint(100000)

        output = scan.exprefixsumNumbaSingle(in_ary, init=init)

        # check last carry
        self.assertTrue(np.isclose(output - init, in_ary_cpy.sum()),
                        'carry return is not sum')

        # check array
        carry = init
        for i in xrange(ARRAY_SIZE):
            self.assertEqual(in_ary[i], carry, 'output array not correct')
            carry += in_ary_cpy[i]


class TestScanGPU(unittest.TestCase):

    def test_scan_int32(self):
        in_ary = np.random.randint(0, 10000, ARRAY_SIZE).astype(np.int32)
        in_ary_d = cuda.to_device(in_ary)

        output = scan.scan_gpu(in_ary_d)

        out_carry = output.getitem(0)
        out_ary = in_ary_d.copy_to_host()

        # check last carry
        self.assertTrue(np.isclose(out_carry, in_ary.sum()),
                        'carry return is not sum')

        # check array
        carry = 0
        for i in xrange(ARRAY_SIZE):
            self.assertEqual(out_ary[i], carry, 'output array not correct')
            carry += in_ary[i]

    def test_scan_int64(self):
        in_ary = np.random.randint(0, 10000, ARRAY_SIZE)
        in_ary_d = cuda.to_device(in_ary)

        output = scan.scan_gpu(in_ary_d)

        out_carry = output.getitem(0)
        out_ary = in_ary_d.copy_to_host()

        # check last carry
        self.assertTrue(np.isclose(out_carry, in_ary.sum()),
                        'carry return is not sum')

        # check array
        carry = 0
        for i in xrange(ARRAY_SIZE):
            self.assertEqual(out_ary[i], carry, 'output array not correct')
            carry += in_ary[i]

    def test_scan_fp32(self):
        in_ary = np.random.rand(ARRAY_SIZE).astype(np.float32)
        in_ary_d = cuda.to_device(in_ary)

        output = scan.scan_gpu(in_ary_d)

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

    def test_scan_fp64(self):
        in_ary = np.random.rand(ARRAY_SIZE).astype(np.float64)
        in_ary_d = cuda.to_device(in_ary)

        output = scan.scan_gpu(in_ary_d)

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
