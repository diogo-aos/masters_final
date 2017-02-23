import numpy as np
import numba as nb

#
# jitted version was 110 times faster than unjitted for 1e6 array
# ported and adapted to arg-k-select from:
# http://blog.teamleadnet.com/2012/07/quick-select-algorithm-find-kth-element.html
@nb.njit
def arg_k_select(ary, k, out):
# def arg_k_select(ary, k):
    args = np.empty(ary.size, dtype=np.int32)
    for i in range(args.size):
        args[i] = i

    fro = 0
    to = ary.size - 1

    while fro < to:
        r = fro
        w = to
        mid_arg = args[(r+w) / 2]
        mid = ary[mid_arg]

        while r < w:
            r_arg = args[r]
            w_arg = args[w]
            if ary[r_arg] >= mid:
                tmp = args[w]
                args[w] = args[r]
                args[r] = tmp
                w -= 1
            else:
                r += 1

        r_arg = args[r]
        if ary[r_arg] > mid:
            r -= 1

        if k <= r:
            to = r
        else:
            fro = r + 1

    for i in range(k):
        out[i] = args[i]

    # return args[:k]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       QUICKSORT
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# from: http://alienryderflex.com/quicksort/
# max_levels=64 should work fine; it only runs into
# problems if the array is at least 2^max_levels
def quicksort(ary_in, max_levels,
              start_idx=None, end_idx=None,
              order="ascending"):

    if order == "ascending":
        quicksort_fn = quicksort_ascending        
    elif order == "descending":
        quicksort_fn = quicksort_descending   
    else:
        raise ValueError("order must be \"ascending\" or \"descending\".")

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = ary.size

    res = quicksort_fn(ary, max_levels, start_idx, end_idx)
    return res

@nb.njit
def quicksort_ascending(ary, max_levels, start_idx, end_idx):
    beg = np.empty(max_levels, dtype=np.int64)
    end = np.empty(max_levels, dtype=np.int64)
    
    beg[0] = start_idx
    end[0] = end_idx
    
    i=0
    while i >= 0:
        L = beg[i]
        R = end[i] - 1
        if L < R:
            piv = ary[L]

            # check if stack is full
            if i == max_levels - 1:
                return -1

            while L < R:
                # search for smaller element
                while ary[R] >= piv and L < R:
                    R -= 1
                if L < R:
                    ary[L] = ary[R]
                    L += 1
                # search for bigger element
                while ary[L] <= piv and L < R:
                    L += 1
                if L < R:
                    ary[R] = ary[L]
                    R -= 1
            ary[L] = piv
            beg[i+1] = L + 1
            end[i+1] = end[i]
            end[i] = L
            i += 1

            # making sure it performs smaller partition first
            if end[i] - beg[i] > end[i-1] - beg[i-1]:
                swap = beg[i]
                beg[i] = beg[i-1]
                beg[i-1] = swap
                
                swap = end[i]
                end[i] = end[i-1]
                end[i-1] = swap

        else:
            i -= 1
    return 0

@nb.njit
def quicksort_descending(ary, max_levels, start_idx, end_idx):
    beg = np.empty(max_levels, dtype=np.int64)
    end = np.empty(max_levels, dtype=np.int64)
    
    beg[0] = start_idx
    end[0] = end_idx
    
    i=0
    while i >= 0:
        L = beg[i]
        R = end[i] - 1
        if L < R:
            piv = ary[L]

            # check if stack is full
            if i == max_levels - 1:
                return -1
                
            while L < R:
                # search for smaller element
                while ary[R] <= piv and L < R:
                    R -= 1
                if L < R:
                    ary[L] = ary[R]
                    L += 1
                # search for bigger element
                while ary[L] >= piv and L < R:
                    L += 1
                if L < R:
                    ary[R] = ary[L]
                    R -= 1
            ary[L] = piv
            beg[i+1] = L + 1
            end[i+1] = end[i]
            end[i] = L
            i += 1
            # making sure it performs smaller partition first
            if end[i] - beg[i] > end[i-1] - beg[i-1]:
                swap = beg[i]
                beg[i] = beg[i-1]
                beg[i-1] = swap
                
                swap = end[i]
                end[i] = end[i-1]
                end[i-1] = swap

        else:
            i -= 1
    return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                         QUICKSORT FOR TWO ARRAYS
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@nb.njit
def quicksort_two(ary, ary2, max_levels,
                  start_idx=None, end_idx=None, order="ascending"):
    if order == "ascending":
        quicksort_fn = quicksort_two_inner_asc        
    elif order == "descending":
        quicksort_fn = quicksort_two_inner_desc   
    else:
        raise ValueError("order must be \"ascending\" or \"descending\".")

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = ary.size

    stack_beg = np.empty(max_levels, dtype=np.int64)
    stack_end = np.empty(max_levels, dtype=np.int64)
    res = quicksort_fn(ary, ary2, stack_beg, stack_end)
    return res

def quicksort_two_inner_asc(ary, ary2, beg, end, start_idx, end_idx):
    """Adapted from http://alienryderflex.com/quicksort/. The stacks are
    received. Sorts ary and changes the order of ary2 according to the new order
    of ary.
    """
    max_levels = beg.size
    
    beg[0] = start_idx
    end[0] = end_idx
    
    i=0
    while i >= 0:
        L = beg[i]
        R = end[i] - 1
        if L < R:
            piv = ary[L]
            piv2 = ary2[L]
            if i == max_levels - 1:
                return -1
            while L < R:
                # search for smaller element
                while ary[R] >= piv and L < R:
                    R -= 1
                if L < R:
                    ary[L] = ary[R]
                    ary2[L] = ary2[R]
                    L += 1
                # search for bigger element
                while ary[L] <= piv and L < R:
                    L += 1
                if L < R:
                    ary[R] = ary[L]
                    ary2[R] = ary2[L]
                    R -= 1
            ary[L] = piv
            ary2[L] = piv2
            beg[i+1] = L + 1
            end[i+1] = end[i]
            end[i] = L
            i += 1
            # making sure it performs smaller partition first
            if end[i] - beg[i] > end[i-1] - beg[i-1]:
                swap = beg[i]
                beg[i] = beg[i-1]
                beg[i-1] = swap
                
                swap = end[i]
                end[i] = end[i-1]
                end[i-1] = swap            
        else:
            i -= 1
    return 0


@nb.njit
def quicksort_two_inner_desc(ary, ary2, beg, end, start_idx, end_idx):
    """Adapted from http://alienryderflex.com/quicksort/. The stacks are
    received. Sorts ary and changes the order of ary2 according to the new order
    of ary.
    """
    max_levels = beg.size
    
    beg[0] = start_idx
    end[0] = end_idx
    
    i=0
    while i >= 0:
        L = beg[i]
        R = end[i] - 1
        if L < R:
            piv = ary[L]
            piv2 = ary2[L]
            if i == max_levels - 1:
                return -1
            while L < R:
                # search for smaller element
                while ary[R] <= piv and L < R:
                    R -= 1
                if L < R:
                    ary[L] = ary[R]
                    ary2[L] = ary2[R]
                    L += 1
                # search for bigger element
                while ary[L] >= piv and L < R:
                    L += 1
                if L < R:
                    ary[R] = ary[L]
                    ary2[R] = ary2[L]
                    R -= 1
            ary[L] = piv
            ary2[L] = piv2
            beg[i+1] = L + 1
            end[i+1] = end[i]
            end[i] = L
            i += 1
            # making sure it performs smaller partition first
            if end[i] - beg[i] > end[i-1] - beg[i-1]:
                swap = beg[i]
                beg[i] = beg[i-1]
                beg[i-1] = swap
                
                swap = end[i]
                end[i] = end[i-1]
                end[i-1] = swap            
        else:
            i -= 1
    return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       CHECK IF ARRAY SORTED
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def check_sorted(ary, start_idx=None, end_idx=None, order="ascending"):
    if order == "ascending":
        check_sorted_fn = check_sorted_asc        
    elif order == "descending":
        check_sorted_fn = check_sorted_desc   
    else:
        raise ValueError("order must be \"ascending\" or \"descending\".")

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = ary.size

    return check_sorted_fn(ary, start_idx, end_idx)

@nb.njit
def check_sorted_asc(ary, start_idx, end_idx):
    """Checks if an array is sorted in the interval [start_idx, end_idx[.
    """
    n = end_idx
    prev_val = ary[start_idx]
    i = start_idx + 1
    
    while i < n:
        new_val = ary[i]
        if new_val < prev_val:
            return -1
        prev_val = new_val
        i += 1
    return 0

@nb.njit
def check_sorted_desc(ary, start_idx, end_idx):
    """Checks if an array is sorted in the interval [start_idx, end_idx[.
    """
    n = end_idx
    prev_val = ary[start_idx]
    i = start_idx + 1
    
    while i < n:
        new_val = ary[i]
        if new_val > prev_val:
            return -1
        prev_val = new_val
        i += 1
    return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                       CUSTOM SORTING FUNCTIONS FOR EAC
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def csr_datasort(data, indices, indptr, max_levels, order="ascending"):
    """This function takes a CSR matrix and will sort the data from each row in
    an increasing order. It will also sort the indices for each row in the same
    order.

    max_levels is the size of the stack to use in the Quick Sort algorithm.

    Returns -i if the sorting was not successful, where i encodes row where the 
    sorting failed - the row is -i+1. This means that the stack was filled and
    needs to be bigger.
    """
    if order == "ascending":
        func = csr_datasort_asc        
    elif order == "descending":
        func = csr_datasort_desc   
    else:
        raise ValueError("order must be \"ascending\" or \"descending\".")

    res = func(data, indices, indptr, max_levels)

    return res

def data_is_sorted(data, indptr, order="ascending"):
    """Checks if all the data subarrays are sorted.
    Returns True if all are sorted and (False,i) if not, where i is the row 
    the sort check failed.
    """
    if order == "ascending":
        check_sorted = check_sorted_asc
    elif order == "descending":
        check_sorted = check_sorted_desc
    else:
         raise ValueError("order must be \"ascending\" or \"descending\".")
       
    for i in range(0,indptr.size-1):
        start = indptr[i]
        end = indptr[i+1]
        if check_sorted(data, start, end) == -1:
            return False, i
    return True

@nb.njit
def csr_datasort_asc(data, indices, indptr, max_levels):
    stack_beg = np.empty(max_levels, dtype=np.int64)
    stack_end = np.empty(max_levels, dtype=np.int64)
    for i in range(indptr.size-1):
        start = indptr[i]
        end = indptr[i+1]
        res = quicksort_two_inner_asc(data, indices,
                                      stack_beg, stack_end,
                                      start, end)
        if res == -1:
            return -i
    return 0

@nb.njit
def csr_datasort_desc(data, indices, indptr, max_levels):
    stack_beg =  np.empty(max_levels, dtype=np.int64)
    stack_end = np.empty(max_levels, dtype=np.int64)
    for i in range(indptr.size-1):
        start = indptr[i]
        end = indptr[i+1]
        res = quicksort_two_inner_desc(data, indices,
                                      stack_beg, stack_end,
                                      start, end)
        if res == -1:
            return -i
    return 0

@nb.njit
def data_argmax(data, indptr, degree):
    arg = -1
    val = -1
    row = -1
    i=0
    n = indptr.size-1
    for i in xrange(i, n):
        i_degree = degree[i]
        if i_degree == 0:
            continue
        i_indptr = indptr[i]
        carg = i_indptr + i_degree - 1
        cval = data[carg]
        if cval > val:
            arg = carg
            val = cval
            row = i
    return arg, row

if __name__ == '__main__':
    x = np.random.randint(0,1000000000,10000000)
    xc = x.copy()
    xc.sort()

    STACK_MAX = 64
    sort_res = quicksort(x, STACK_MAX)

    print "sort res:", sort_res
    print x
    print xc

    assert np.all(xc == x)
    print('all ok')
