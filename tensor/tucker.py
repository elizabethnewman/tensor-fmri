import numpy
import numpy as np
from numpy.testing import assert_array_equal
from numpy.linalg import eig, norm
from tensor.mode_k import modek_unfold, modek_fold, modek_product
from copy import copy, deepcopy


# https://gitlab.com/tensors/tensor_toolbox/-/blob/master/hosvd.m

def hosvd(X, tol, ranks=None, sequential=False, dim_order=None):
    # || X - T || / || X || <= tol
    ndim = X.ndim
    shape_X = X.shape

    if dim_order is None:
        dim_order = np.arange(ndim)

    if ranks is None:
        ranks = np.zeros(ndim)

    assert len(dim_order) == ndim, "Order of dimensions must be of length ndims(X)"
    assert_array_equal(np.sort(dim_order), np.arange(ndim))
    assert len(ranks) == ndim, "Order of ranks must be of length ndims(X)"

    nrm_X2 = np.sum(X ** 2)
    eigsumthresh = (tol ** 2) * nrm_X2 / ndim

    Y = deepcopy(X)
    U = ndim * [None]
    for k in dim_order:
        Yk = modek_unfold(Y, k)
        d, u = eig(Yk @ Yk.T)

        idx = np.argsort(d)
        idx = np.flip(idx)

        # sort
        d = d[idx]

        if ranks[k] < 1:
            # cumulative sum of the smallest singular values, squared
            eigsum = np.cumsum(np.flip(d))

            ranks[k] = shape_X[k] - np.sum(eigsum < eigsumthresh)

            if ranks[k] < 1:
                ranks[k] = 1  # must take at least the first singular value

        U[k] = u[:, idx[:int(ranks[k])]]

        if sequential:
            Y = modek_product(Y, U[k].T, k)

    # form core
    G = deepcopy(Y)
    if not sequential:
        for k in dim_order:
            G = modek_product(G, U[k].T, k)

    return G, U, ranks


def tucker_product(G, U, transpose=False, dim_order=None):

    X = modek_product(G, U, axis=dim_order, transpose=transpose)

    return X





