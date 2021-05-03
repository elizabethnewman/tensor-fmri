import numpy as np
from numpy.linalg import svd
from numpy.testing import assert_array_almost_equal
from tensor.utils import assert_compatile_sizes_facewise, facewise, facewise_t_svd, facewise_t_svdII
from scipy.fft import fft, ifft
from math import prod
import time

# ==================================================================================================================== #
# transforms
def t_fft(A):
    # apply one-dimensional fft along last 2 through d dimensions of A
    ndim = np.ndim(A)

    for i in range(2, ndim):
        A = fft(A, axis=i, norm='ortho')

    return A


def t_ifft(A):
    # apply one-dimensional ifft along last d through 2 dimensions of A (order doesn't matter)
    ndim = np.ndim(A)
    for i in range(ndim - 1, 1, -1):
        A = ifft(A, axis=i, norm='ortho')

    # ensure A is real-valued (a somewhat expensive, but necessary check)
    assert_array_almost_equal(np.imag(A), np.zeros_like(A))
    A = np.real(A)

    return A

# ==================================================================================================================== #
# t-product
def t_product(A, B):

    assert_compatile_sizes_facewise(A, B)

    # move to transform domain
    A_hat = t_fft(A)
    B_hat = t_fft(B)

    # compute facewise product
    C_hat = facewise(A_hat, B_hat)

    # return to spatial comain
    C = t_ifft(C_hat)

    return C


def t_product_eye(shape_I):
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


def t_transpose(A):
    A = np.swapaxes(A, 0, 1)
    ndim = A.ndim

    for i in range(2, ndim):
        A = t_flip_hold_dim0(A, i)

    # slower option
    # flip2 = lambda x, axis: t_flip_hold_dim0(x, axis)
    # start = time.time()
    # A = np.apply_over_axes(flip2, A, np.arange(2, A.ndim))
    # end = time.time()
    # print(end - start)

    return A


def t_flip_hold_dim0(A, axis):
    # TODO: reduce copying
    A0 = np.expand_dims(np.take(A, 0, axis=axis), axis=axis)
    A2 = np.take(A, np.arange(1, A.shape[axis]), axis=axis)
    A2 = np.flip(A2, axis=axis)

    return np.concatenate((A0, A2), axis=axis)


# ==================================================================================================================== #
def t_svd(A, k=None):
    # A = U * fdiag(S) * VH

    # transform
    A = t_fft(A)

    U, s, VH = facewise_t_svd(A, k)

    # return to spatial domain
    U = t_ifft(U)
    S = t_ifft(np.reshape(s, (1, *s.shape)))[0]  # remove first dimension
    VH = t_ifft(VH)

    return U, S, VH


def t_svdII(A, gamma, compress_UV=False, return_spatial=True):
    # A = U * fdiag(S) * VH

    # transform
    A = t_fft(A)
    # nrm_Ahat = np.linalg.norm(A)

    U, S, VH, multi_rank = facewise_t_svdII(A, gamma, compress_UV=compress_UV)

    # return to spatial domain
    if return_spatial:
        U = t_ifft(U)
        S = t_ifft(np.reshape(S, (1, *S.shape)))[0]   # remove first dimension
        VH = t_ifft(VH)

    return U, S, VH, multi_rank



