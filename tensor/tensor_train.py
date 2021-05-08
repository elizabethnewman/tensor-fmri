import numpy
import numpy as np
from numpy.testing import assert_array_equal
from numpy.linalg import svd, norm
from tensor.mode_k import modek_unfold, modek_fold, modek_product
from tensor.utils import reshape
from copy import copy, deepcopy


# https://github.com/oseledets/TT-Toolbox/blob/master/%40tt_tensor/tt_tensor.m
def ttsvd(X, tol, ranks=None, dim_order=None):
    # ranks is length ndim

    ndim = X.ndim
    shape_X = X.shape

    if dim_order is None:
        dim_order = np.arange(ndim)

    assert len(dim_order) == ndim, "Order of dimensions must be of length ndims(X)"
    assert_array_equal(np.sort(dim_order), np.arange(ndim))

    # store ranks
    r = np.ones(ndim + 1)

    C = deepcopy(np.transpose(X, dim_order))
    G = ndim * [None]

    ep = tol / np.sqrt(ndim - 1)

    for i in range(ndim - 1):
        m = int(shape_X[i] * r[i])
        C = reshape(C, [m, -1])

        u, s, vh = svd(C, compute_uv=True, full_matrices=False)

        if ranks is None:
            r1 = chop(s, ep * norm(s))
        else:
            r1 = ranks[i]

        r[i + 1] = r1

        # truncate
        s = s[:r1]
        u = u[:, :r1]
        vh = vh[:r1, :]

        # store
        G[i] = u

        # form new C
        C = np.diag(s) @ vh

    G[-1] = C

    return G, r


def chop(s, tol):

    if np.isclose(norm(s), np.zeros(1)):
        r = 1
        return r

    r_max = s.size
    if tol <= 0:
        return r_max

    s0 = np.cumsum(np.flip(s) ** 2)
    ff = np.argwhere(s0 < tol ** 2)
    r = r_max - len(ff)

    return r


def tt_product(G, shape_X, dim_order=None):
    ndim = len(G)

    if dim_order is None:
        dim_order = np.arange(ndim)

    assert len(dim_order) == ndim, "Order of dimensions must be of length ndims(X)"
    assert_array_equal(np.sort(dim_order), np.arange(ndim))

    Xk = G[ndim - 1]

    r = list(map(lambda x: x.shape[1], G))
    for i in range(ndim - 2, -1, -1):
        u = G[i]

        Xk = u @ Xk

        if i > 0:
            Xk = reshape(Xk, [r[i - 1], -1])

    shape_X = np.array(shape_X)
    Xk = reshape(Xk, list(shape_X[dim_order]))

    Xk = np.transpose(Xk, list(np.argsort(dim_order)))

    return Xk

