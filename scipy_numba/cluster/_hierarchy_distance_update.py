"""
A `linkage_distance_update` function calculates the distance from cluster i
to the new cluster xy after merging cluster x and cluster y

Parameters
----------
d_xi : double
    Distance from cluster x to cluster i
d_yi : double
    Distance from cluster y to cluster i
d_xy : double
    Distance from cluster x to cluster y
size_x : int
    Size of cluster x
size_y : int
    Size of cluster y
size_i : int
    Size of cluster i

Returns
-------
d_xyi : double
    Distance from the new cluster xy to cluster i
"""

import math

import numba as nb

sig = nb.double(nb.double, nb.double, nb.double, nb.int64, nb.int64, nb.int64)


@nb.njit(sig)
def _single(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    return min(d_xi, d_yi)


@nb.njit(sig)
def _complete(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    return max(d_xi, d_yi)


@nb.njit(sig)
def _average(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    return (size_x * d_xi + size_y * d_yi) / (size_x + size_y)


@nb.njit(sig)
def _centroid(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    return math.sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                     (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                     (size_x + size_y))


@nb.njit(sig)
def _median(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    return math.sqrt(0.5 * (d_xi * d_xi + d_yi * d_yi) - 0.25 * d_xy * d_xy)


@nb.njit(sig)
def _ward(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    t = 1.0 / (size_x + size_y + size_i)
    return math.sqrt((size_i + size_x) * t * d_xi * d_xi +
                     (size_i + size_y) * t * d_yi * d_yi -
                      size_i * t * d_xy * d_xy)


@nb.njit(sig)
def _weighted(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    return 0.5 * (d_xi + d_yi)


if __name__ == '__main__':
    import numpy as np
    x = np.asarray([1, 2, 3], dtype=np.double)
    n = np.asarray([1, 2, 3], dtype=np.int64)
    print(_single(x[0], x[1], x[2], n[0], n[1], n[2]))
    print(_complete(x[0], x[1], x[2], n[0], n[1], n[2]))
    print(_average(x[0], x[1], x[2], n[0], n[1], n[2]))
    print(_centroid(x[0], x[1], x[2], n[0], n[1], n[2]))
    print(_median(x[0], x[1], x[2], n[0], n[1], n[2]))
    print(_ward(x[0], x[1], x[2], n[0], n[1], n[2]))
    print(_weighted(x[0], x[1], x[2], n[0], n[1], n[2]))
