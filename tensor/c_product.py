import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal
from tensor.utils import assert_compatile_sizes_facewise, reshape
import tensor.f_product as fprod
from scipy.fft import dct, idct


def c_transform(A, norm='ortho'):
    # apply one-dimensional fft along last 2 through d dimensions of A
    ndim = np.ndim(A)

    for i in range(2, ndim):
        A = dct(A, axis=i, norm=norm)

    return A


def c_itransform(A, norm='ortho'):
    # apply one-dimensional ifft along last d through 2 dimensions of A (order doesn't matter)
    ndim = np.ndim(A)
    for i in range(ndim - 1, 1, -1):
        A = idct(A, axis=i, norm=norm)

    return A

# ==================================================================================================================== #
# c-product
def c_prod(A, B):

    assert_compatile_sizes_facewise(A, B)

    # move to transform domain
    A_hat = c_transform(A)
    B_hat = c_transform(B)

    # compute facewise product
    C_hat = fprod.f_prod(A_hat, B_hat)

    # return to spatial comain
    C = c_itransform(C_hat)

    # ensure C is real-valued
    assert_array_almost_equal(np.imag(C), np.zeros_like(C))
    C = np.real(C)

    return C


def c_eye(shape_I):
    assert shape_I[0] == shape_I[1], "Identity tensor must have square frontal slices"

    # create identity tube
    id_tube_hat = np.ones([1, 1, *shape_I[2:]])
    id_tube = c_itransform(id_tube_hat)

    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = id_tube

    return I


def c_tran(A):

    return np.swapaxes(A, 0, 1)


# ==================================================================================================================== #
def c_svd(A, k=None):
    # A = U * fdiag(S) * VH

    shape_A = A.shape

    # transform
    A = c_transform(A)

    U, s, VH, stats = fprod.f_svd(A, k)
    stats['nrm_A'] = norm(A)

    # return to spatial domain
    U = c_itransform(U)
    S = reshape(c_itransform(np.reshape(s, (1, *s.shape))), (s.shape[0], *shape_A[2:]))  # remove first dimension
    VH = c_itransform(VH)

    return U, S, VH, stats


def c_svdII(A, gamma, compress_UV=True, return_spatial=True, implicit_rank=None):
    # A = U * fdiag(S) * VH

    # transform
    A = c_transform(A)

    U, S, VH, stats = fprod.f_svdII(A, gamma, compress_UV=compress_UV, implicit_rank=implicit_rank)
    stats['nrm_A'] = norm(A)
    # return to spatial domain
    if return_spatial:
        U = c_itransform(U)
        S = c_itransform(reshape(S, (1, *S.shape)))[0]   # remove first dimension
        VH = c_itransform(VH)

    return U, S, VH, stats