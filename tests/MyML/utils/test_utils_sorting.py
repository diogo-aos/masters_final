import numpy as np
import MyML.utils.sorting as MySort

def test_quicksort_ascending():
    x = np.random.randint(0, 1000000, 1000000)
    xc = x.copy()
    xc.sort()  # sort with numpy func

    sort_res = MySort.quicksort(x, order="ascending")

    print "sort res:", sort_res
    print x
    print xc

    assert np.all(xc == x)

def test_quicksort_descending():
    x = np.random.randint(0, 1000000, 1000000)
    xc = x.copy()
    xc.sort()  # sort with numpy func
    xc = xc[::-1]

    sort_res = MySort.quicksort(x, order="descending")

    print "sort res:", sort_res
    print x
    print xc

    assert np.all(xc == x)

def test_sort_2arys_asc(): raise NotImplementedError


def test_sort_2arys_desc(): raise NotImplementedError


def test_k_select(): raise NotImplementedError


def test_csr_datasort_asc(): raise NotImplementedError


def test_csr_datasort_desc(): raise NotImplementedError
