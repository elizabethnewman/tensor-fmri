import numpy as np
from numpy.linalg import svd, norm
from tensor.utils import assert_compatile_sizes_facewise, reshape
import math


# ==================================================================================================================== #
def f_transform(A):
    return A


def f_itransform(A):
    return A


# ==================================================================================================================== #
# facewise
def f_prod(A, B):
    """
    Multiply the frontal slice of tensors of compatible size.
    Parameters
    ----------
    A : (p x n1 x n2 x ... x nd) array
    B : (n1 x m x n2 x ... x nd) array

    Returns
    -------
    C : (p x m x n2 x ... x nd) array
    """
    assert_compatile_sizes_facewise(A, B)

    # reorient for python batch multiplication
    A = np.moveaxis(A, [0, 1], [-2, -1])
    B = np.moveaxis(B, [0, 1], [-2, -1])

    # facewise multiply
    C = A @ B

    # return to matlab orientation
    C = np.moveaxis(C, [-2, -1], [0, 1])

    return C


# ==================================================================================================================== #
# general purpose functions
def f_eye(shape_I):
    # identity tensor is the same in the transform domain
    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = 1.0

    return I


def f_tran(A):
    return np.swapaxes(A, 0, 1)


# ==================================================================================================================== #
def f_svd(A, k=None):
    # default, but have to form full tensors u and vh before truncating
    # could be too expensive with larger data, and a loop may be preferable

    # determine sizes
    if k is None:
        k = min(A.shape[0], A.shape[1])
    else:
        k = min(k, min(A.shape[0], A.shape[1]))

    A = np.moveaxis(A, [0, 1], [-2, -1])
    u, s, vh = svd(A, full_matrices=False)

    u = np.moveaxis(u, [-2, -1], [0, 1])
    u = u[:, :k]

    s = np.moveaxis(s, -1, 0)
    s = s[:k]

    vh = np.moveaxis(vh, [-2, -1], [0, 1])
    vh = vh[:k]

    # store any useful statistics
    stats = dict()
    nrm_A = norm(A)
    stats['rank'] = k
    stats['nrm_Ahat'] = nrm_A
    stats['nrm_A'] = nrm_A
    stats['err'] = np.sqrt(np.abs(nrm_A ** 2 - np.sum(s ** 2)))
    stats['rel_err'] = stats['err'] / stats['nrm_A']
    stats['total_storage'] = u.size + vh.size
    stats['compression_ratio'] = A.size / stats['total_storage']

    return u, s, vh, stats


def f_svdII(A, gamma, compress_UV=False, implicit_rank=None):

    # compute full t-svd
    U, s, VH, svd_stats = f_svd(A)
    nrm_A = svd_stats['nrm_A']

    # truncate singular values
    s = f_svdII_cutoff(s, gamma, implicit_rank=implicit_rank)
    multi_rank = np.sum(reshape(s, (s.shape[0], -1)) != 0, axis=0)
    implicit_rank = np.count_nonzero(s)

    if compress_UV:
        U, s, VH = f_svdII_compress_UV(U, s, VH, multi_rank)

    stats = dict()
    stats['nrm_Ahat'] = nrm_A
    stats['implicit_rank'] = implicit_rank
    stats['multi_rank'] = multi_rank
    stats['err'] = np.sqrt(nrm_A ** 2 - np.sum(s ** 2))
    stats['rel_err'] = stats['err'] / nrm_A
    stats['total_storage'] = implicit_rank * (A.shape[0] + A.shape[1])
    stats['compression_ratio'] = A.size / stats['total_storage']

    return U, s, VH, stats


def f_svdII_cutoff(s, gamma, implicit_rank=None):
    if implicit_rank is None:
        if gamma is not None and gamma < 1:
            d2 = np.flip(np.sort(reshape(s, -1) ** 2))
            idx = np.argwhere(np.cumsum(d2) / np.sum(d2) < gamma)

            if len(idx) < 1:
                idx = 0
            else:
                idx = idx[-1]

            cutoff = np.sqrt(d2[idx])
            s[s < cutoff] = 0
    else:
        shape_s = s.shape
        s = reshape(s, -1)
        idx = np.flip(np.argsort(s ** 2))
        cutoff_idx = idx[implicit_rank:]
        s[cutoff_idx] = 0
        s = reshape(s, shape_s)

    return s


def f_svdII_compress_UV(U, s, VH, multi_rank):

    m = np.max(multi_rank)
    U = U[:, :m]
    s = s[:m]
    VH = VH[:m, :]

    # # the correct way to do this, but problems with complex arithmetic
    # shape_U = U.shape
    # shape_s = s.shape
    # shape_VH = VH.shape
    # U = reshape(U, (U.shape[0], U.shape[1], -1))
    # s = reshape(s, (s.shape[0], -1))
    # VH = reshape(VH, (VH.shape[0], VH.shape[1], -1))
    #
    # for i in range(min(multi_rank), max(multi_rank) + 1):
    #     U[:, i:, multi_rank <= i] = 0
    #     s[i:, multi_rank <= i] = 0
    #     VH[i:, :, multi_rank <= i] = 0
    #
    # U = reshape(U, shape_U)
    # s = reshape(s, shape_s)
    # VH = reshape(VH, shape_VH)

    return U, s, VH
