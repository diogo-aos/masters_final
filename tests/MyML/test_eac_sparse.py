from numpy import load, uint8, float32

import MyML.EAC.sparse as spEAC
from MyML.cluster import KMeans

# load data
iris = load('iris.npy', )
data = iris[:,:4]
labels = iris[:,4].astype(uint8)

# generate partition
generator = KMeans()
