"""
TODO:
 - checks in squareform; change default value to True
"""


import numpy as np
import numba as nb

def squareform(X, force="no", checks=False):
    """
    Converts a vector-form distance vector to a square-form distance
    matrix, and vice-versa.
    Parameters
    ----------
    X : ndarray
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to 'tovector' or 'tomatrix',
        the input will be treated as a distance matrix or distance vector
        respectively.
    checks : bool, optional
        If `checks` is set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero.
        These values are ignored any way so they do not disrupt the
        squareform transformation.
    Returns
    -------
    Y : ndarray
        If a condensed distance matrix is passed, a redundant one is
        returned, or if a redundant one is passed, a condensed distance
        matrix is returned.
    Notes
    -----
    1. v = squareform(X)
       Given a square d-by-d symmetric distance matrix X,
       ``v=squareform(X)`` returns a ``d * (d-1) / 2`` (or
       `${n \\choose 2}$`) sized vector v.
      v[{n \\choose 2}-{n-i \\choose 2} + (j-i-1)] is the distance
      between points i and j. If X is non-square or asymmetric, an error
      is returned.
    2. X = squareform(v)
      Given a d*d(-1)/2 sized v for some integer d>=2 encoding distances
      as described, X=squareform(v) returns a d by d distance matrix X. The
      X[i, j] and X[j, i] values are set to
      v[{n \\choose 2}-{n-i \\choose 2} + (j-u-1)] and all
      diagonal elements are zero.
    """

    dtype = X.dtype
    s = X.shape

    if force.lower() == 'tomatrix':
        if len(s) != 1:
            raise ValueError("Forcing 'tomatrix' but input X is not a "
                             "distance vector.")
    elif force.lower() == 'tovector':
        if len(s) != 2:
            raise ValueError("Forcing 'tovector' but input X is not a "
                             "distance matrix.")


    if len(s) == 1:
        if X.shape[0] == 0:
            return np.zeros((1, 1), dtype=dtype)

        # Grab the closest value to the square root of the number
        # of elements times 2 to see if the number of elements
        # is indeed a binomial coefficient.
        d = int(np.ceil(np.sqrt(X.shape[0] * 2)))

        # Check that v is of valid dimensions.
        if d * (d - 1) / 2 != int(s[0]):
            raise ValueError('Incompatible vector size. It must be a binomial '
                             'coefficient n choose 2 for some integer n >= 2.')

        # Allocate memory for the distance matrix.
        M = np.empty((d, d), dtype=dtype)

        # Fill in the values of the distance matrix.
        to_squareform_from_vector(M, X)

        # Return the distance matrix.
        return M

    elif len(s) == 2:
        if s[0] != s[1]:
            raise ValueError('The matrix argument must be square.')
        if checks:
            raise NotImplementedError("checks are not implemented")
            is_valid_dm(X, throw=True, name='X')

        # One-side of the dimensions is set here.
        d = s[0]

        if d <= 1:
            return np.array([], dtype=dtype)

        # Create a vector.
        v = np.empty((d * (d - 1)) // 2, dtype=dtype)

        # Convert the vector to squareform.
        to_vector_from_squareform(X, v)
        return v        

    else:
        raise ValueError(('The first argument must be one or two dimensional '
                         'array. A %d-dimensional array is not '
                         'permitted') % len(s))        


@nb.njit
def to_vector_from_squareform(mat, vec):
    n = mat.shape[0]
    idx = 0
    for i in range(n-1):
        for j in range(i+1, n):
            vec[idx] = mat[i,j]
            idx += 1

@nb.njit
def to_squareform_from_vector(mat, vec):
    n = mat.shape[0]
    idx = 0
    for i in range(n-1):
        mat[i,i] = 0
        for j in range(i+1, n):
            val = vec[idx]
            mat[i,j] = val
            mat[j,i] = val
            idx += 1
    mat[j,j] = 0


if __name__ == '__main__':
    squareform_test()


def squareform_test():
    n=134
    c_size = (n * (n-1)) // 2
    c = np.arange(c_size, dtype=np.int16)

    f = squareform(c)
    assert f.dtype == c.dtype

    cn = squareform(f)
    assert np.all(cn == c)
    assert c.dtype == cn.dtype

    print('squareform: all ok')