import numpy as np
from ..utils.partition import convertIndexToBin as convertIndexToPos
from ..utils.partition import convertClusterStringToBin as convertClusterStringToPos

import munkres # module with Hungarian (Munkres) algorithm


class HungarianIndex():

    def __init__(self, nsamples):

        self.N = nsamples

    def score(self, clusts1, clusts2, format='array',
              format1=None, format2=None):

        if format1 == None:
            format1 = format

        if format2 == None:
            format2 = format

        if format1 == 'list':
            clusts1_ = convertIndexToPos(clusts=clusts1,N=self.N)
        elif format1 == 'array':
            clusts1_ = convertClusterStringToPos(clusts=clusts1,N=self.N)
        elif format1 == 'bin':
            clusts1_ = clusts1
        else:
            raise ValueError("Format not accepted: {}".format(format1))

        if format2 == 'list':
            clusts2_ = convertIndexToPos(clusts=clusts2, N=self.N)
        elif format2 == 'array':
            clusts2_ = convertClusterStringToPos(clusts=clusts2, N=self.N)
        elif format2 == 'bin':
            clusts2_ = clusts2            
        else:
            raise ValueError("Format not accepted: {}".format(format2))

        nclusts1_ = clusts1_.shape[0]
        nclusts2_ = clusts2_.shape[0]

        match_matrix = clusts1_.dot(clusts2_.T)

        self.clusts1_ = clusts1_
        self.clusts2_ = clusts2_        

        # pad matrix to square
        match_matrix = pad_to_square(match_matrix)

        # convert cost matrix to profit
        profit_matrix = match_matrix.max() + 1 - match_matrix

        # apply Munkres algorithm
        m = munkres.Munkres()
        indexes = m.compute(profit_matrix)

        match_count = 0
        for row, col in indexes:
            match_count += match_matrix[row, col]

        self.match_count = match_count
        self.accuracy = np.float(self.match_count) / self.N

        return self.accuracy


def pad_to_square(mat):
    if mat.ndim != 2:
        raise TypeError("Array must have two dimensions")

    nrows, ncols = mat.shape

    dif = nrows - ncols

    # matrix already square
    if dif == 0:
        return mat

    # more rows than cols, pad cols
    if dif > 0:
        return np.pad(mat, ((0,0),(0,dif)), mode='constant', constant_values=0)

    # more cols than rows, pad rows
    if dif < 0:
        return np.pad(mat, ((0,-dif),(0,0)), mode='constant', constant_values=0)