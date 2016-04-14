import numpy as np

from numba import jit, njit
from scipy.sparse import csr_matrix

from MyML.utils.scan import exprefixsumNumbaSingle as ex_prefix_sum

import tables # store matrix in file

#from MyML.cluster.linkage import binary_search, binary_search_interval

class EAC_CSR():

    def __init__(self, n_samples, max_assocs, max_assocs_type="constant",
                 dtype=np.uint8, condensed=False, sort_mode="numpy",
                 **kwargs):
        """
        Inputs:
            n_samples       : number of samples (integer)
            max_assocs      : maximum number of associations (integer)
            max_assocs_type : scheme to use for the maximum number of assocs of
                              each sample
                \"constant\" - (default) same num. assocs for all samples
                \"linear\" - linear cut on the number of assocs
                If matrix is not condensed then the type must be\"constant\".                              
            dtype           : type of data (integer or float NumPy type)
            condensed       : fill only upper triangular matrix (boolean)
            sort_mode       : method to use when sorting the columns of rows
                \"numpy\"  - (default) uses NumPy quicksort routine
                \"normal\" - uses faster scheme

        If max_assocs_type=="linear" than 4 extra arguments may be passed:
        n_s(0.5)    : where the linear cut should start in percentage relative
                      to the number of samples
        n_e(1.0)    : where the linear cut should end in percentage relative
                      to the number of samples
        val_s(1.0)  : start value for the number of associations in percentage
                      relative to the maximum number of associations supplied
        val_e(0.05) : end value for the number of associations in percentage
                      relative to the maximum number of associations supplied
        E.g. n_s=0.05, n_e=0.95, val_s=1.0, val_e=0.05,
             n_samples=100, max_assocs=20
             The interval of samples [0,4] will have 20 assocs,
             the interval [5,94] will have assocs according to the line 
             connecting points (5,20) to (95,1) (y=-0.211(1)x + 21.05556),
             the interval [95,99] will have 1 assoc

        """
        self.n_samples = n_samples
        self.max_assocs = max_assocs
        self.max_assocs_type = max_assocs_type
        self.dtype = dtype

        self.condensed = condensed
        self.sort_mode = sort_mode

        self._validate_func_options()

        # linear indptr args
        self.n_s = kwargs.get("n_s", 0.05)
        self.n_e = kwargs.get("n_e", 1.0)
        self.val_s = kwargs.get("val_s", 1.0)
        self.val_e = kwargs.get("val_e", 0.05)

        self._allocate_arrays()

        self.nnz = 0
        self.numpy_sort = False

        self.update_partition = self._update_first_partition

        self.csr_form = False
        self.nnz_discarded = 0

        self.is_diassoc = False

        # the matrix will usually be created with the knowledge of the
        # number of partitions
        self.max_val = kwargs.get("max_val", None)

    def _allocate_arrays(self):
        n_samples = self.n_samples
        max_assocs = self.max_assocs

        self.degree = np.zeros(n_samples + 1, dtype=np.int32)

        if self.max_assocs_type == "constant":
            self.indptr = indptr_constant(self.n_samples, self.max_assocs)
        elif self.max_assocs_type == "linear":
            self.indptr = indptr_linear(self.n_samples, self.max_assocs,
                                        self.n_s, self.n_e,
                                        self.val_s, self.val_e)

        total_assocs = self.indptr[-1]
        
        self.indices = np.empty(total_assocs, dtype=np.int32)
        self.data = np.zeros(total_assocs, dtype=self.dtype)

        # store total size in bytes
        data_bytes = self.data.size * self.data.itemsize
        indices_bytes = self.indices.size * self.indices.itemsize
        indptr_bytes = self.indptr.size * self.indptr.itemsize
        degree_bytes = self.degree.size * self.degree.itemsize

        self.alloc_size = (data_bytes + indices_bytes 
                               + indptr_bytes + degree_bytes)
        

    def _validate_func_options(self):
        if self.condensed not in (True,False):
            raise TypeError("condensed must be boolean")

        if type(self.max_assocs) != int:
            raise TypeError("max_assocs must be an integer")
        if self.max_assocs <= 0:
            raise ValueError("max_assocs must be an integer above 0")

        if type(self.n_samples) != int:
            raise TypeError("number of samples must be an integer")
        if self.n_samples <= 0:
            raise ValueError("number of samples must be an integer above 0")

        if not issubclass(self.dtype, (np.integer, np.float)):
            raise TypeError("dtype must be a NumPy integer or float type")           

        sort_options = ("numpy", "normal", "simple", "surgical")
        if self.sort_mode not in sort_options:
            raise ValueError("sort_mode must be one of the "
                             "following: {}".format(sort_options))

        max_assocs_options = ("constant","linear")
        if self.max_assocs_type not in max_assocs_options:
            raise ValueError("max_assocs_type must be one of the "
                             "following: {}".format(max_assocs_options))

        if self.max_assocs_type != "constant" and not self.condensed:
            raise ValueError("max_assocs_type must be \"constant\" if matrix "
                             "is condensed")

    def _set_mode(self):
        self._validate_func_options()

        if not hasattr(self,"fp_fn"):
            self.fp_fn = update_funcs["first"][self.condensed]

        if not hasattr(self,"normal_fn"):
            self.normal_fn = update_funcs["normal"][self.condensed][self.sort_mode]

        if self.sort_mode == "numpy":
            self.numpy_sort = True
        else:
            self.numpy_sort = False

    def update_ensemble(self, ensemble, progress=False):
        # choose the first partition to be the one with least clusters (more 
         # samples per cluster)
        first_partition = np.argmin(map(len,ensemble))
        self.update_partition(ensemble[first_partition])
        for p in xrange(len(ensemble)):
            if p == first_partition:
                continue

            # if progress:
            #     yield p

            self.update_partition(ensemble[p])

    def _update_first_partition(self, partition):
        self._set_mode()

        for cluster in partition:
            nnnz = self.fp_fn(self.indices, self.data, self.indptr,
                                 self.degree, cluster)
            self.nnz += nnnz

        if self.numpy_sort:
            self.update_partition = self._update_partition_numpy
        else:
            self.update_partition = self._update_partition_fast

    def _update_partition_numpy(self, partition):
        for cluster in partition:
            nnnz, dnnz = self.normal_fn(self.indices, self.data, self.indptr,
                                        self.degree, cluster, self.max_assocs)
            self.nnz += nnnz
            self.nnz_discarded += dnnz
        sort_indices_numpy(self.indices, self.data, self.indptr,
                           self.degree, self.n_samples)
       
    def _update_partition_fast(self, partition):
        for cluster in partition:
            nnnz, dnnz = self.normal_fn(self.indices, self.data, self.indptr,
                                        self.degree, cluster, self.max_assocs)
            self.nnz += nnnz
            self.nnz_discarded += dnnz

    def todense(self):
        if not self.csr_form:
            self._condense()

        return self.csr.todense()

    def tocsr(self, copy=False):
        if not self.csr_form:
            self._condense()

        if copy:
            return self.csr.copy()
        else:
            return self.csr

    def _condense(self, keep_degree = False):
        nnz = self.nnz
        condense_eac_csr(self.indices, self.data, self.indptr, self.degree)
        self.indices = self.indices[:nnz]
        self.data = self.data[:nnz]
        self.indptr[-1] = nnz

        self.csr_form = True
        self.csr = csr_matrix((self.data, self.indices, self.indptr),
                              shape=(self.n_samples, self.n_samples))
        # self.csr.eliminate_zeros()

        if not keep_degree:
            del self.degree



    def store(self, file, delete=False, indptr_expanded=True,
              store_degree=False, chunkshape=None):
        """TODO: if indptr is not to be expanded save as chunked arrays.
        file    : (string) pathname to where the matrix should be saved;
                  if path points to existing file, it will be overwritten.
        delete  : (boolean) deletes the memory co-association matrix after 
                  saving it do disk.
        """

        if store_degree and not hasattr(self, "degree"):
            raise AttributeError("degree doesn't exist.")

        if not indptr_expanded:
            raise NotImplementedError("Saving normal indptr not implemented.")

        #create hdf5 file
        with tables.openFile(file, 'w') as f: 
        
            NE = self.nnz
            N = self.n_samples
            max_assocs = self.max_assocs

            # determine datatypes
            data_dtype = self.data.dtype
            col_ind_dtype = self.indices.dtype
            if indptr_expanded:
                row_ind_dtype = self.indices.dtype
            else:
                self.indptr.dtype

            #create description of your table
            class Table_Description(tables.IsDescription):
                data = tables.Col.from_dtype(data_dtype, pos=0)
                col_ind = tables.Col.from_dtype(col_ind_dtype, pos=1)
                row_ind = tables.Col.from_dtype(row_ind_dtype, pos=2)

            #create table
            a = f.create_table("/","graph", 
                               description=Table_Description,
                               expectedrows=NE, chunkshape=chunkshape)
            a._v_attrs.N = N
            a._v_attrs.NE = NE
            a._v_attrs.max_assocs = max_assocs

            data = self.data
            col_ind = self.indices
            indptr = self.indptr

            for row in xrange(N):
                start = indptr[row]
                end = indptr[row+1]
                a.append((data[start:end],
                          col_ind[start:end],
                          np.full(end-start, row, dtype=row_ind_dtype)))

            if store_degree:
                datatype = self.degree.dtype
                atom = tables.Atom.from_dtype(datatype)
                filters = tables.Filters(complib='blosc', complevel=5)
                ds = f.createCArray('/', 'degree', atom,
                                    self.degree.shape, filters=filters)
                ds[:] = self.degree

        if delete:
            del self.data, self.indices, self.indptr, data, col_ind, indptr

    def make_diassoc(self, max_val=None):
        if max_val is None:
            if self.max_val is not None:
                max_val = self.max_val + 1
            else:
                max_val = self.data.max() + 1

        make_diassoc_1d(self.data, max_val)

        self.is_diassoc = True

#1d
@njit
def make_diassoc_1d(ary, val):
    for i in range(ary.size):
        tmp = ary[i]
        ary[i] = val - tmp


def load_coassoc(file):
    raise NotImplementedError("Not implemeted loading a coassoc.")
    #create hdf5 file
    with tables.openFile(file, 'w') as f:

        a = f.root.graph

        N = a._v_attrs.N
        NE = a._v_attrs.NE
        max_assocs = a._v_attrs.max_assocs

        col_ind = a.read('col')








# degree will be the indptr of the condensed CSR
#
@njit
def condense_eac_csr(indices, data, indptr, degree):
    ptr = degree[0]
    n_samples = degree.size - 1
    indptr[0] = 0
    for i in range(1, n_samples):
        i_ptr = indptr[i]
        stopcond = i_ptr + degree[i]
        indptr[i] = ptr
        while i_ptr < stopcond:
            indices[ptr] = indices[i_ptr]
            data[ptr] = data[i_ptr]
            ptr += 1
            i_ptr += 1

def sort_indices_numpy(indices, data, indptr, degree, n_samples):
    # sort all rows by indices
    for row in xrange(n_samples):
        start_i = indptr[row] # start index
        # start_i = row * max_assocs # deduced start index
        end_i = start_i + degree[row] # end index
        asorted = indices[start_i:end_i].argsort() # get sorted order

        # update data and indices with sorted order
        data[start_i:end_i] = data[start_i:end_i][asorted]
        indices[start_i:end_i] = indices[start_i:end_i][asorted]



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                 Strategies for number of assocs per samples
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def indptr_constant(n_samples, max_assocs):
    return np.arange(n_samples + 1, dtype=np.int64) * max_assocs

def indptr_linear(n_samples, max_assocs, n_s, n_e, val_s, val_e):
    """
    This function will return the indptr using a linear model for the maximum
    number of association of each sample. This function will take O(2*n_samples)
    space.
    Inputs:
        n_samples       : number of samples
        max_assocs      : maximum number of associations
        n_s             : start of the cut in percentage relative to n_sample
        n_e             : end of the cut in percentage relative to n_sample
        pc_s            : initial value in percentage relative to max_assocs
        pc_e            : end value in percentage relative to max_assocs
    Output:
        indptr          : array of size n_samples+1 where the i-th element is 
                          the start index of the i-th sample. The difference
                          between any two consecutive elements (i and i+1) is 
                          the number of associations of the sample with index i.
                          The last element is the total number of associations.
    """

    x = np.arange(n_samples, dtype=np.int32)
    y = np.empty(n_samples + 1, dtype=np.int64)

    n_s = n_s * n_samples # where to start to cut
    n_e = n_e * n_samples # where to end the cut
    val_s = val_s * max_assocs # start value
    val_e = val_e * max_assocs # end value

    m = (val_e - val_s) / (n_e - n_s) 
    b = val_s - n_s * m

    n_s = int(n_s)
    n_e = int(n_e)
    val_s = int(val_s)
    val_e = int(val_e)

    y[:n_s] = val_s
    y[n_s:n_e] = m * x[n_s:n_e] + b
    y[n_e:] = val_e

    ex_prefix_sum(y)

    return y

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                 update cluster of first partition
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@njit
def update_cluster_fp(indices, data, indptr, degree, cluster):
    nnz = 0

    for i in range(cluster.size):
        n = cluster[i] # sample id
        fa = indptr[n] # index of first assoc

        add_ptr = 0
        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            # add association
            data[fa + add_ptr] = 1
            indices[fa + add_ptr] = na
            add_ptr += 1
        degree[n] = add_ptr
        nnz += add_ptr


    return nnz

@njit
def update_cluster_fp_condensed(indices, data, indptr, degree, cluster):
    nnz = 0

    for i in range(cluster.size):
        n = cluster[i] # sample id
        fa = indptr[n] # index of first assoc

        add_ptr = 0
        for j in range(i+1, cluster.size):

            na = cluster[j] # sample id of association
            
            # add association
            data[fa + add_ptr] = 1
            indices[fa + add_ptr] = na
            add_ptr += 1
        degree[n] = add_ptr
        nnz += add_ptr

    return nnz

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# update cluster of any partition; discards associations that exceed
# pre-allocated space for each sample (=max_assocs)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
@njit
def update_cluster(indices, data, indptr, degree, cluster, max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0
    discarded_nnz = 0

    for i in range(cluster.size):
        n = cluster[i] # sample id
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + degree[n] # index of last assoc
        new_n_degree = degree[n]
        discarded_nnz = 0

        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue            
            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    discarded_nnz += 1
                    continue

                # update number of associations of n
                new_n_degree += 1                

                # index to add new association
                new_assoc_ind = fa + new_n_degree - 1

                data[new_assoc_ind] = 1
                indices[new_assoc_ind] = na

        # update number of new non zero elements
        nnz += new_n_degree - degree[n]

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz, discarded_nnz

@njit
def update_cluster_condensed(indices, data, indptr, degree,
                             cluster, max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0
    discarded_nnz = 0

    for i in range(cluster.size - 1):
        n = cluster[i] # sample id
        fa = indptr[n] # index of first assoc
        la = fa + degree[n] # index of last assoc
        new_n_degree = degree[n]

        max_assocs = indptr[n+1] - fa # max_assocs for n
        discarded_nnz = 0

        for j in range(i+1, cluster.size):
            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    discarded_nnz += 1
                    continue

                # update number of associations of n
                new_n_degree += 1                

                # index to add new association
                new_assoc_ind = fa + new_n_degree - 1

                data[new_assoc_ind] = 1
                indices[new_assoc_ind] = na

        # update number of new non zero elements
        nnz += new_n_degree - degree[n]

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz, discarded_nnz

@njit
def update_cluster_sorted(indices, data, indptr, degree, cluster,
                          max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0
    discarded_nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int64)
    new_assocs_idx = np.empty(max_assocs, dtype=np.int64)

    for i in range(cluster.size):
        n = cluster[i] # sample id
        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        new_assocs_ptr = 0
        discarded_nnz = 0

        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    discarded_nnz += 1
                    continue

                new_assocs_ids[new_assocs_ptr] = na
                new_assocs_idx[new_assocs_ptr] = -(ind + 1)

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

        ## make sorted
        # sort
        n_ptr = new_assocs_ptr - 1
        i_ptr = fa + (n_degree - 1)
        o_ptr = fa + n_degree + new_assocs_ptr - 1
        last_index = i_ptr

        while o_ptr >= fa:
            if n_ptr < 0:
                indices[o_ptr] = indices[i_ptr]
                data[o_ptr] = data[i_ptr]
                o_ptr -= 1
                i_ptr -= 1
                continue

            idx = new_assocs_idx[n_ptr]

            # insert new assocs at end
            if idx > last_index:
                indices[o_ptr] = new_assocs_ids[n_ptr]
                data[o_ptr] = 1
                o_ptr -= 1
                n_ptr -= 1
            # add original assocs
            elif i_ptr >= idx:
                
                # try:
                #     indices[o_ptr] = indices[i_ptr]
                # except:
                #     print "i:", i
                #     print "optr:", o_ptr
                #     print "iptr:", i_ptr
                #     raise Exception
                indices[o_ptr] = indices[i_ptr]
                data[o_ptr] = data[i_ptr]
                o_ptr -= 1
                i_ptr -= 1
            # add new assoc
            else:
                indices[o_ptr] = new_assocs_ids[n_ptr]
                data[o_ptr] = 1
                o_ptr -= 1
                n_ptr -= 1

    return nnz, discarded_nnz

@njit
def update_cluster_sorted_condensed(indices, data, indptr, degree,
                                    cluster, max_assocs):

    # maximum number of associations pre-allocated for each sample
    nnz = 0
    discarded_nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int64)
    new_assocs_idx = np.empty(max_assocs, dtype=np.int64)
    # new_assocs_idx_f = np.empty(max_assocs - new_n_degree, dtype=np.int32)

    for i in range(cluster.size - 1):
        n = cluster[i] # sample id
        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        max_assocs = indptr[n+1] - fa # max_assocs for n

        new_assocs_ptr = 0
        discarded_nnz = 0

        for j in range(i+1, cluster.size):
            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    discarded_nnz += 1
                    continue

                new_assocs_ids[new_assocs_ptr] = na
                new_assocs_idx[new_assocs_ptr] = -(ind + 1)

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        ## make sorted
        # sort
        n_ptr = new_assocs_ptr - 1
        i_ptr = fa + (n_degree - 1)
        o_ptr = fa + n_degree + new_assocs_ptr - 1
        last_index = i_ptr

        while o_ptr >= fa:
            if n_ptr < 0:
                indices[o_ptr] = indices[i_ptr]
                data[o_ptr] = data[i_ptr]
                o_ptr -= 1
                i_ptr -= 1
                continue

            idx = new_assocs_idx[n_ptr]

            # insert new assocs at end
            if idx > last_index:
                indices[o_ptr] = new_assocs_ids[n_ptr]
                data[o_ptr] = 1
                o_ptr -= 1
                n_ptr -= 1
            # add original assocs
            elif i_ptr >= idx:
                indices[o_ptr] = indices[i_ptr]
                data[o_ptr] = data[i_ptr]
                o_ptr -= 1
                i_ptr -= 1
            # add new assoc
            else:
                indices[o_ptr] = new_assocs_ids[n_ptr]
                data[o_ptr] = 1
                o_ptr -= 1
                n_ptr -= 1

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz, discarded_nnz

@njit
def update_cluster_sorted_simple(indices, data, indptr, degree, cluster, max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0
    discarded_nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int64)

    for i in range(cluster.size):
        n = cluster[i] # sample id
        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        new_assocs_ptr = 0
        discarded_nnz = 0

        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    discarded_nnz += 1
                    continue

                new_assocs_ids[new_assocs_ptr] = na

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        ## make sorted

        # sort
        n_ptr = new_assocs_ptr - 1
        i_ptr = fa + (n_degree - 1)
        o_ptr = fa + n_degree + new_assocs_ptr - 1
        last_index = i_ptr

        n_ptr_id = new_assocs_ids[n_ptr]
        i_ptr_id = indices[i_ptr]
        i_ptr_data = data[i_ptr]

        while o_ptr >= fa:

            # second condition for when all new assocs have been added
            # and only old ones remain
            if i_ptr_id > n_ptr_id or n_ptr < 0:
                indices[o_ptr] = i_ptr_id
                data[o_ptr] = i_ptr_data
                i_ptr -= 1
                i_ptr_id = indices[i_ptr]
                i_ptr_data = data[i_ptr]
            else:
                indices[o_ptr] = n_ptr_id
                data[o_ptr] = 1
                n_ptr -= 1
                if n_ptr >= 0:
                    n_ptr_id = new_assocs_ids[n_ptr]
            o_ptr -= 1

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz, discarded_nnz

#
# surgical because there are no branches while sorting
#
@njit
def update_cluster_sorted_surgical(indices, data, indptr, degree, cluster,
                                   max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0
    discarded_nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int64)
    new_assocs_idx = np.empty(max_assocs, dtype=np.int64)

    for i in range(cluster.size):
        n = cluster[i] # sample id
        if n > 405109:
            return nnz, discarded_nnz

        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        new_assocs_ptr = 0

        
        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0: 
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    discarded_nnz += 1
                    continue

                new_assocs_ids[new_assocs_ptr] = na
                new_assocs_idx[new_assocs_ptr] = -(ind + 1)

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

        ## make sorted

        # shift original indices
        new_assocs_idx[new_assocs_ptr] = fa + n_degree
        n_shifts = new_assocs_ptr
        o_ptr = fa + new_assocs_ptr + n_degree - 1
        while n_shifts >= 1:
            start_idx = new_assocs_idx[n_shifts - 1]
            end_idx = new_assocs_idx[n_shifts] - 1

            # shift original
            while end_idx >= start_idx:
                indices[o_ptr] = indices[end_idx]
                data[o_ptr] = data[end_idx]
                end_idx -= 1
                o_ptr -= 1

            #add new
            indices[o_ptr] = new_assocs_ids[n_shifts - 1]
            data[o_ptr] = 1
            o_ptr -= 1            

            n_shifts -= 1

    return nnz, discarded_nnz

#
# surgical because there are no branches while sorting
#
@njit
def update_cluster_sorted_surgical_condensed(indices, data, indptr, degree,
											 cluster, max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0
    discarded_nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int64)
    new_assocs_idx = np.empty(max_assocs, dtype=np.int64)

    for i in range(cluster.size-1):
        n = cluster[i] # sample id
        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        max_assocs = indptr[n+1] - fa # max_assocs for n

        new_assocs_ptr = 0

        for j in range(i+1,cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    discarded_nnz += 1
                    continue

                new_assocs_ids[new_assocs_ptr] = na
                new_assocs_idx[new_assocs_ptr] = -(ind + 1)

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

        ## make sorted

        # shift original indices
        new_assocs_idx[new_assocs_ptr] = fa + n_degree
        n_shifts = new_assocs_ptr
        o_ptr = fa + new_assocs_ptr + n_degree - 1
        while n_shifts >= 1:
            start_idx = new_assocs_idx[n_shifts - 1]
            end_idx = new_assocs_idx[n_shifts] - 1

            # shift original
            while end_idx >= start_idx:
                indices[o_ptr] = indices[end_idx]
                data[o_ptr] = data[end_idx]
                end_idx -= 1
                o_ptr -= 1

            #add new
            indices[o_ptr] = new_assocs_ids[n_shifts - 1]
            data[o_ptr] = 1
            o_ptr -= 1            

            n_shifts -= 1

    return nnz, discarded_nnz

# first level key (first, normal) choose between first partition or others
# second level (True,False) chooses if matrix is condensed or not
# third level (numpy,normal,simple,surgical) chooses sorting method for other
# partitions

first_part_funcs = {True : update_cluster_fp_condensed,
                    False: update_cluster_fp}

normal_part_funcs = {True:{"numpy":update_cluster_condensed,
                           "normal":update_cluster_sorted_condensed,
                           "surgical":update_cluster_sorted_surgical_condensed},
                    False:{"numpy":update_cluster,
                           "normal":update_cluster_sorted,
                           "simple":update_cluster_sorted_simple,
                           "surgical":update_cluster_sorted_surgical}}

update_funcs = {"first": first_part_funcs,
                "normal": normal_part_funcs}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                  UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def _compute_max_assocs_from_ensemble(ensemble):
    return max([max(map(np.size,p)) for p in ensemble])


@njit
def binary_search_interval(key, ary, start, end):
    """
    Inputs:
        key         : value to find
        ary         : sorted arry in which to find the key
        start, end  : interval of the array in which to perform the search
    Outputs:
        if the search was successful the output is a positive number with the
        index of where the key exits in ary; if not the output is a negative
        number; the symmetric of that number plus 1 is the index of where the
        key would be inserted in such a way that the array would still be sorted

    """
    imin = start
    imax = end

    while imin < imax:
        imid = (imax + imin) / 2
        imid_val = ary[imid]

        # key is before
        if key < imid_val:
            imax = imid
        # key is after
        elif key > imid_val:
            imin = imid + 1
        # key is between first edge of imid and next first edge
        else:
            return imid
    return -imin - 1