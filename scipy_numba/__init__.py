"""
=========================================
Clustering package (:mod:`scipy.cluster`)
=========================================

.. currentmodule:: scipy.cluster

:mod:`scipy.cluster.vq`

Clustering algorithms are useful in information theory, target detection,
communications, compression, and other areas.  The `vq` module only
supports vector quantization and the k-means algorithms.

:mod:`scipy.cluster.hierarchy`

The `hierarchy` module provides functions for hierarchical and
agglomerative clustering.  Its features include generating hierarchical
clusters from distance matrices,
calculating statistics on clusters, cutting linkages
to generate flat clusters, and visualizing clusters with dendrograms.

"""
"""
from __future__ import division, print_function, absolute_import

#__all__ = ['vq', 'hierarchy']

#from . import vq, hierarchy

from . import hierarchy

from numpy.testing import Tester
test = Tester().test
"""
