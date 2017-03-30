# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:52:13 2015

@author: Diogo Silva


#TODO:
- add exception handling
"""

import pickle

def readPickle(filename):
    """
    Receives a filename to read from, opens the file, loads the pickled data 
    inside to memory and returns an object with that data.
    """

    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data
    
def writePickle(filename,data):
    """
    Receives a filename to write to and the data to write, opens the file, 
    writes the data to the file.
    """
    
    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()