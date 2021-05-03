import numpy as np
from numpy.testing import assert_array_almost_equal
from tensor.utils import assert_compatile_sizes_facewise, facewise, facewise_t_svd, facewise_t_svdII
from scipy.fft import dct, idct

def t_dct(A, norm='ortho'):
    # apply one-dimensional fft along last 2 through d dimensions of A
    ndim = np.ndim(A)

    for i in range(2, ndim):
        A = dct(A, axis=i, norm=norm)

    return A


def t_idct(A, norm='ortho'):
    # apply one-dimensional ifft along last d through 2 dimensions of A (order doesn't matter)
    ndim = np.ndim(A)
    for i in range(ndim - 1, 1, -1):
        A = idct(A, axis=i, norm=norm)

    return A

# ==================================================================================================================== #
# c-product
def c_product(A, B):

    assert_compatile_sizes_facewise(A, B)

    # move to transform domain
    A_hat = t_dct(A)
    B_hat = t_dct(B)

    # compute facewise product
    C_hat = facewise(A_hat, B_hat)

    # return to spatial comain
    C = t_idct(C_hat)

    # ensure C is real-valued
    assert_array_almost_equal(np.imag(C), np.zeros_like(C))
    C = np.real(C)

    return C


def c_product_eye(shape_I):
    assert shape_I[0] == shape_I[1], "Identity tensor must have square frontal slices"

    # create identity tube
    id_tube_hat = np.ones([1, 1, *shape_I[2:]])
    id_tube = t_idct(id_tube_hat)

    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = id_tube

    return I


def c_transpose(A):

    return np.swapaxes(A, 0, 1)

# ==================================================================================================================== #
def c_svd(A, k=None):
    # A = U * fdiag(S) * VH

    shape_A = A.shape

    # transform
    A = t_dct(A)

    U, s, VH = facewise_t_svd(A, k)

    # return to spatial domain
    U = t_idct(U)
    S = np.reshape(t_idct(np.reshape(s, (1, *s.shape))), (s.shape[0], *shape_A[2:]))  # remove first dimension
    VH = t_idct(VH)

    return U, S, VH


def c_svdII(A, gamma, compress_UV=True, return_spatial=True):
    # A = U * fdiag(S) * VH

    # transform
    A = t_dct(A)
    # nrm_Ahat = np.linalg.norm(A)

    U, S, VH, multi_rank = facewise_t_svdII(A, gamma, compress_UV=compress_UV)

    # return to spatial domain
    if return_spatial:
        U = t_idct(U)
        S = t_idct(np.reshape(S, (1, *S.shape)))[0]   # remove first dimension
        VH = t_idct(VH)

    return U, S, VH, multi_rank