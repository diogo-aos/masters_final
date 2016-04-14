# -*- coding: utf-8 -*-
"""
Created on: 15-09-2015
Last modified: 15-09-2015

@author: Diogo Silva

This file contains rule functions for computing 

TODO:

"""

import numpy as np

# rules for picking kmin kmax 
def rule1(n):
    """sqrt"""
    k = [np.sqrt(n)/2, np.sqrt(n)]
    k = map(np.ceil,k)
    k = map(int, k)
    return k

def rule2(n):
    """2sqrt"""
    k =  map(lambda x:x*2,rule1(n))
    return k

def rule3(n, sk, th):
    """fixed s/k"""
    k = [n * 1.0 / sk, th * n * 1.0 / sk]
    k = map(np.ceil, k)
    k = map(int, k)
    return k

def rule4(n):
    """sk=sqrt_2,th=30%"""
    return rule3(n, sk1(n), 1.3)

def rule5(n):
    """sk=300,th=30%"""
    return rule3(n, 300, 1.3)

def rule6(n, th=1.3):
    '''sqrt * log10, th=30%'''
    kmin = np.sqrt(n) * np.log10(n) / 2
    kmax = kmin * th
    return map(int, [kmin, kmax])

# rules for picking number of samples per cluster
def sk1(n):
    """sqrt/2"""
    return int(np.sqrt(n) / 2)