# Dependencies
## Quantum Clustering
 - NumPy
 - SciKit-Learn
 - bitstring

## GPU SL
 - Numpy
 - Numba
 - NumbaPro (only for Single-Link)

## EAC
 - Numpy
 - Numba
 - SciPy
 - scipy_numba (included in repo)
 - PyTables (for very large sparse matrices)

# Usage
Make sure scipy_numba and this repository are in the PYTHONPATH, so that the code is availale from anywhere.


# TODO
## Dev
 - Float support in GPU Scan

## Tests
 - scan (DONE)
 - EAC
 - Boruvka CPU
 - Boruvka GPU
 - SL CPU
 - SL GPU
 - graph build
 - graph mst
 - graph connected components