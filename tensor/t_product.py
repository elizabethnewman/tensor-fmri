import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal
import tensor.f_product as fprod
from tensor.utils import assert_compatile_sizes_facewise, reshape
from scipy.fft import fft, ifft
# from math import prod
from utils.general_utils import prod

# ==================================================================================================================== #
# transforms
def t_transform(A):
    # apply one-dimensional fft along last 2 through d dimensions of A
    ndim = np.ndim(A)

    for i in range(2, ndim):
        A = fft(A, axis=i, norm='ortho')

    return A


def t_itransform(A):
    # apply one-dimensional ifft along last d through 2 dimensions of A (order doesn't matter)
    ndim = np.ndim(A)
    for i in range(ndim - 1, 1, -1):
        A = ifft(A, axis=i, norm='ortho')

    # ensure A is real-valued (a somewhat expensive, but useful check)
    # assert_array_almost_equal(np.imag(A), np.zeros_like(A))
    A = np.real(A)

    return A


# ==================================================================================================================== #
# t-product
def t_prod(A, B):

    assert_compatile_sizes_facewise(A, B)

    # move to transform domain
    A_hat = t_transform(A)
    B_hat = t_transform(B)

    # compute facewise product
    C_hat = fprod.f_prod(A_hat, B_hat)

    # return to spatial comain
    C = t_itransform(C_hat)

    return C


def t_eye(shape_I):
    # takes in a tuple or list of sizes

    assert shape_I[0] == shape_I[1], "Identity tensor must have square frontal slices"

    # create identity tube
    id_tube = np.zeros([1, 1, *shape_I[2:]])

    # for 'ortho' (default)
    np.put(id_tube, np.zeros(len(shape_I)).astype(int), np.sqrt(prod(shape_I[2:])))

    # for 'backward'
    # np.put(id_tube, np.zeros(len(shape_I)).astype(int), 1.0)

    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = id_tube

    return I


def t_tran(A):
    A = np.swapaxes(A, 0, 1)
    ndim = A.ndim

    for i in range(2, ndim):
        A = t_flip_hold_dim0(A, i)

    return A


def t_flip_hold_dim0(A, axis):
    A0 = np.expand_dims(np.take(A, 0, axis=axis), axis=axis)
    A2 = np.take(A, np.arange(1, A.shape[axis]), axis=axis)
    A2 = np.flip(A2, axis=axis)

    return np.concatenate((A0, A2), axis=axis)


# ==================================================================================================================== #
def t_svd(A, k=None):
    # A = U * fdiag(S) * VH

    # transform
    A = t_transform(A)

    U, s, VH, stats = fprod.f_svd(A, k)
    stats['nrm_A'] = norm(A)

    # return to spatial domain
    U = t_itransform(U)
    S = t_itransform(np.reshape(s, (1, *s.shape)))[0]  # remove first dimension
    VH = t_itransform(VH)

    return U, S, VH, stats


def t_svdII(A, gamma, compress_UV=False, return_spatial=True, implicit_rank=None):
    # A = U * fdiag(S) * VH

    # transform
    A = t_transform(A)
    # nrm_Ahat = np.linalg.norm(A)

    U, S, VH, stats = fprod.f_svdII(A, gamma, compress_UV=compress_UV, implicit_rank=implicit_rank)
    stats['nrm_A'] = norm(A)

    # return to spatial domain
    if return_spatial:
        U = t_itransform(U)
        S = t_itransform(np.reshape(S, (1, *S.shape)))[0]   # remove first dimension
        VH = t_itransform(VH)

    return U, S, VH, stats



