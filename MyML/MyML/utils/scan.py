# -*- coding: utf-8 -*-
"""

TODO:
 - fix bug for input array of size 1 - just zero out the array (of len 1)
 - optimize scan for bank conflicts - how to do this in Python Numba CUDA?
 - make scan generic, i.e. accept both device and host arrays
 - make GPU version work with floats -> currently doesn't work
"""


import numpy as np
import numba
from numba import cuda, int32, float32, void
from timeit import default_timer as timer


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                                 SEQUENTIAL
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def exprefixsum(masks, indices, init=0, nelem=None):
    """
    exclusive prefix sum
    """
    nelem = masks.size if nelem is None else nelem

    carry = init
    for i in xrange(nelem):
        indices[i] = carry
        if masks[i] != 0:
            carry += masks[i]

    #indices[nelem] = carry
    return carry


@numba.njit
def exprefixsumNumba(in_ary, out_ary, init=0):
    """
    exclusive prefix sum
    """
    nelem = in_ary.size

    carry = init
    for i in range(nelem):
        out_ary[i] = carry
        carry += in_ary[i]

    return carry


@numba.njit
def exprefixsumNumbaSingle(in_ary, init=0):
    """
    exclusive prefix sum
    """
    nelem = in_ary.size

    carry = init
    keeper = in_ary[0]
    in_ary[0] = init
    for i in range(1, nelem):
        carry += keeper
        keeper = in_ary[i]
        in_ary[i] = carry

    carry += keeper  # total sum
    return carry

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                                 CUDA
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@cuda.jit
def advanced_scan(g_odata, g_idata, n, aux):
    """
    Bleloch algorithm.
    receives auxiliary array to store the whole sum
    only works for array of max size 1024
    adapted to Numba CUDA from
        [1] M. Harris, S. Sengupta, and J. D. Owens,
        \“Parallel Prefix Sum (Scan) with CUDA Mark,\” Gpu gems 3, no. April, pp. 1–24, 2007.

    """
    temp = cuda.shared.array(shape = 0, dtype = numba.i4)

    thid = cuda.threadIdx.x # thread id in block
    tgid = cuda.grid(1) # thread id in grid
    bid = cuda.blockIdx.x # block id

    # load input into shared memory
    temp[2 * thid] = g_idata[2 * thid]
    temp[2 * thid + 1] = g_idata[2 * thid + 1]

    offset = 1

    # build sum in place up the tree
    d = n / 2
    while d > 0:
        cuda.syncthreads()

        if thid < d:
            ai = offset * (2 * thid + 1) - 1
            bi = offset * (2 * thid + 2) - 1

            temp[bi] += temp[ai]
        offset <<= 1 # multipy by 2
        d >>= 1 # divide by 2

    # clear the last element
    if thid == 0:
        temp[n - 1] = 0

    # traverse down tree and build scan
    d = 1
    while d < n:
        offset >>= 1
        cuda.syncthreads()

        if thid < d:
            ai = offset * (2 * thid + 1) - 1
            bi = offset * (2 * thid + 2) - 1

            t = temp[ai]
            temp[ai] = temp[bi]
            temp[bi] += t

        d *= 2

    cuda.syncthreads()

    # write results to device memory
    g_odata[2 * thid] = temp[2 * thid]
    g_odata[2 * thid + 1] = temp[2 * thid + 1]


def scan_gpu(in_ary, MAX_TPB=512, stream=0):
    '''
    Performs an exclusive prefix sum (scan) recursively using GPU with CUDA.
        in_ary  : input array
        MAX_TPB : maximum number of CUDA threads per block; default of 512
        stream  : CUDA stream to use; one is created if none is provided
    Only works with integer arrays currently.
    '''

    n = in_ary.size
    work_ary = in_ary

    tpb = MAX_TPB  # number of threads per block
    epb = tpb * 2  # number of elements per block

    # number of whole blocks, if 0 only 1 incomplete block to process
    bpg = n // epb
    # num. of elements to process in last block, if 0 no last block to process
    elb = n % epb

    if isinstance(in_ary, np.ndarray):
        work_ary = cuda.to_device(in_ary)

    # if not isinstance(work_ary, cuda.cudadrv.devicearray.DeviceNDArray):
    #     raise TypeError("input array must be a device array")

    if not np.issubdtype(work_ary.dtype, int):
        raise NotImplementedError('non integer arrays are not supported')

    # if there is only one block
    if bpg == 0 or (bpg == 1 and elb == 0):
        # minimum power of 2 to include num. of elements to in last block
        p2elb = np.int(np.ceil(np.log2(elb)))
        # total number of elements in last block (counting with extra 0s)
        telb = 2 ** p2elb
        # total number of threads per block
        tlb = telb / 2

        # start index of last block; 0 because it's the only block
        startIdx = 0

        # size of shared memory = telb
        sm_size = telb * work_ary.dtype.itemsize

        # store only the final sum
        dAux = cuda.device_array(shape=1, dtype=work_ary.dtype,
                                 stream=stream)
        auxidx = 0

        last_scan[1, tlb, stream, sm_size](work_ary, dAux, auxidx, elb,
                                           startIdx)

        final_output = dAux
        if isinstance(in_ary, np.ndarray):
            final_output = dAux.getitem(0)

        return final_output
    # if more than one block
    else:
        # number of scans is equal to the number of blocks plus the last block
        n_scans = bpg
        if elb != 0:  # if there is last block
            n_scans += 1

        # +1 because we want the total sum as a side result
        dAux = cuda.device_array(shape=n_scans, dtype=work_ary.dtype,
                                 stream=stream)

        # shared memory is of the size of the elements of block
        sm_size = epb * work_ary.dtype.itemsize

        # prescan all the whole blocks
        prescan[bpg, tpb, stream, sm_size](work_ary, dAux)

        # prescan the last block, if any
        if elb != 0:
            # minimum power of 2 to include elb
            p2elb = np.int(np.ceil(np.log2(elb)))
            # total number of elements in last block (counting with extra 0s)
            telb = 2 ** p2elb
            # total number of threads per block
            tlb = telb / 2

            # start index of last block; 0 because it's the only block
            startIdx = 0

            # size of shared memory = telb
            sm_size = telb * work_ary.dtype.itemsize

            # index of where to save sum of last block
            auxidx = n_scans - 1

            # index of first element of last block
            startIdx = n - elb

            last_scan[1, tlb, stream, sm_size](work_ary, dAux, auxidx,
                                               elb, startIdx)

        # if n_scans is less than maximum number of elements per block
        # it's the last scan
        total_sum = scan_gpu(dAux, stream=stream)

        # sum kernel
        scan_sum[n_scans, tpb, stream](work_ary, dAux)

        return total_sum


# @cuda.jit("void(int32[:], int32[:])")
@cuda.jit
def scan_sum(g_data, aux):
    temp = cuda.shared.array(shape=1, dtype=numba.i4)

    thid = cuda.threadIdx.x  # thread id in block
    bid = cuda.blockIdx.x  # block id

    if thid == 0:
        temp[0] = aux[bid]

    tgid = cuda.grid(1)  # thread id in grid
    elid = tgid * 2  # each thread processes 2 elements

    n = g_data.size

    if elid >= n:
        return

    # synchronize to make sure value to sum is loaded in memory
    cuda.syncthreads()

    g_data[elid] += aux[bid]  # do the sum

    if elid + 1 < n:
        g_data[elid + 1] += aux[bid]


# @cuda.jit("void(int32[:], int32[:])")
@cuda.jit
def prescan(g_data, aux):
    """
    Performs the Bleloch scan.
    Assumes blocks part of a larger array. Sum of block saved in auxiliary
    array. These sums are used to compute the scan of the final, larger array.
    """
    temp = cuda.shared.array(shape=0, dtype=int32)

    thid = cuda.threadIdx.x  # thread id in block
    tgid = cuda.grid(1)  # thread id in grid
    bid = cuda.blockIdx.x  # block id

    bsize = cuda.blockDim.x

    # load input into shared memory
    temp[2 * thid] = g_data[2 * tgid]
    temp[2 * thid + 1] = g_data[2 * tgid + 1]

    offset = 1

    # build sum in place up the tree
    d = bsize
    while d > 0:
        cuda.syncthreads()

        if thid < d:
            ai = offset * (2 * thid + 1) - 1
            bi = offset * (2 * thid + 2) - 1

            temp[bi] += temp[ai]
        offset <<= 1  # multipy by 2
        d >>= 1  # divide by 2

    # save sum to sums array and clear last element
    if thid == 0:
        # the last element processed by this block is the size
        # of the block multiplied by 2
        last_elem_idx = bsize * 2 - 1
        aux[bid] = temp[last_elem_idx]
        temp[last_elem_idx] = 0

    # traverse down tree and build scan
    d = 1
    while d < bsize << 1:
        offset >>= 1
        cuda.syncthreads()

        if thid < d:
            ai = offset * (2 * thid + 1) - 1
            bi = offset * (2 * thid + 2) - 1

            t = temp[ai]
            temp[ai] = temp[bi]
            temp[bi] += t

        d <<= 1

    cuda.syncthreads()

    # write results to device memory, in global IDs
    g_data[2 * tgid] = temp[2 * thid]
    g_data[2 * tgid + 1] = temp[2 * thid + 1]


# @cuda.jit("void(int32[:], int32[:], int32, int32, int32)")
@cuda.jit
def last_scan(g_data, aux, auxidx, elb, start_idx):
    """
    Performs the Bleloch scan on last block, where size may vary.
    g_data : array to perform scan on
    aux : where to store sum
    auxidx : where to store sum in aux array; if auxid == -1 it means that this is not part of
             a large array scan and sums should not be stored
    elb : number of elements of last block
    """
    temp = cuda.shared.array(shape=0, dtype=int32)

    thid = cuda.threadIdx.x  # thread id in block
    tgid = cuda.grid(1)  # thread id in grid
    bid = cuda.blockIdx.x  # block id

    bsize = cuda.blockDim.x

    # load input into shared memory
    # if index is above number of elements in last block,
    # shared memory should be 0
    idx1 = 2 * thid
    idx2 = 2 * thid + 1

    if idx1 < elb:
        temp[idx1] = g_data[start_idx + idx1]
    else:
        temp[idx1] = 0

    if idx2 < elb:
        temp[idx2] = g_data[start_idx + idx2]
    else:
        temp[idx2] = 0

    offset = 1

    # build sum in place up the tree
    d = bsize  # bsize is half the number of elements to process
    while d > 0:
        # if thid == 0:
        #     from pdb import set_trace; set_trace()
        cuda.syncthreads()

        if thid < d:
            ai = offset * (2 * thid + 1) - 1
            bi = offset * (2 * thid + 2) - 1

            temp[bi] += temp[ai]
        offset <<= 1  # multipy by 2
        d >>= 1  # divide by 2

    # clear the last element
    if thid == 0:

        # the last element processed by this block is the size
        # of the block multiplied by 2
        last_elem_id = bsize * 2 - 1

        if auxidx != -1:
            aux[auxidx] = temp[last_elem_id]

        temp[last_elem_id] = 0

    # traverse down tree and build scan
    d = 1
    while d < bsize << 1:  # same thing as before
        offset >>= 1
        cuda.syncthreads()

        if thid < d:
            ai = offset * (2 * thid + 1) - 1
            bi = offset * (2 * thid + 2) - 1

            t = temp[ai]
            temp[ai] = temp[bi]
            temp[bi] += t

        d <<= 1

    cuda.syncthreads()

    # write results to device memory, in global IDs
    if idx1 < elb:
        g_data[start_idx + idx1] = temp[idx1]
    if idx2 < elb:
        g_data[start_idx + idx2] = temp[idx2]
