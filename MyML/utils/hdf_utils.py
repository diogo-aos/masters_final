import tables
import numpy as np

def save_arrays_to_hdf(filename, array_dict, filters=None):
	"""Saves a dictionary of arrays to a file with HDF format. Each array will
	be saved in a HFD node with the name specified its respective key.

	filename 	: name of file
	array_dict 	: a dictionary where the key is the name of the array and the
				  value is the array itself.
	filters 	: default is blosc5
	"""
	f = tables.open_file(filename, 'w')

	if filters is None:
		filters = tables.Filters(complib='blosc', complevel=5)

	for ary_name, ary in array_dict.iteritems():
		atom = tables.Atom.from_dtype(ary.dtype)
		hdf_ary = f.create_carray(f.root, str(ary_name), atom, ary.shape,
								  filters=filters)
		hdf_ary[:] = ary[:]

	f.close()