import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal
from tensor.utils import assert_compatile_sizes_facewise, reshape
import tensor.f_product as fprod
from scipy.fft import dct, idct


# choose dimensions to input (this will yield an error if input exceeds number of dimensions)
# dim_list=(2,3)
# str_=str(input("Input dimensions: "))
# list_=str_.split(" ")
# for i in list_:
# dim_list += (int(i),)


def cf_transform(A, dim_list, norm='ortho'):
    # apply one-dimensional fft along n-th through k-th dimensions of A

    if max(dim_list) >= np.ndim(A):
        raise ValueError("Dimensions do not match")

    for i in dim_list:
        A = dct(A, axis=i, norm=norm)

    return A

    # dim_list=helper_dims(A)


def cf_itransform(A,dim_list, norm='ortho'):
    # apply one-dimensional ifft along n-th through k-th dimensions of A (order doesn't matter)
    # dim_list = helper_dims(A)
    if max(dim_list) >= np.ndim(A):
        raise ValueError("Dimensions do not match")
    for i in dim_list[::-1]:
        A = idct(A, axis=i, norm=norm)

    return A


def cf_prod(A, B, dim_list):
    assert_compatile_sizes_facewise(A, B)

    # dim_list = helper_dims(A)
    # move to transform domain
    A_hat = cf_transform(A, dim_list)
    B_hat = cf_transform(B, dim_list)

    # compute facewise product
    C_hat = fprod.f_prod(A_hat, B_hat)

    # return to spatial comain
    C = cf_itransform(C_hat, dim_list)

    # ensure C is real-valued
    assert_array_almost_equal(np.imag(C), np.zeros_like(C))
    C = np.real(C)

    return C


def cf_eye(shape_I, dim_list):
    assert shape_I[0] == shape_I[1], "Identity tensor must have square frontal slices"

    # create identity tube
    id_tube_hat = np.ones([1, 1, *shape_I[2:]])
    id_tube = cf_itransform(id_tube_hat, dim_list)

    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = id_tube

    return I


def cf_tran(A):
    return np.swapaxes(A, 0, 1)


def cf_svd(A, dim_list, k=None):
    # A = U * fdiag(S) * VH

    shape_A = A.shape

    # transform
    A = cf_transform(A, dim_list)

    U, s, VH, stats = fprod.f_svd(A, k)
    stats['nrm_A'] = norm(A)

    # return to spatial domain
    U = cf_itransform(U, dim_list)
    S = reshape(cf_itransform(np.reshape(s, (1, *s.shape)), dim_list),
                (s.shape[0], *shape_A[2:]))  # remove first dimension
    VH = cf_itransform(VH, dim_list)

    return U, S, VH, stats


def cf_svdII(A, gamma, dim_list, compress_UV=True, return_spatial=True, implicit_rank=None):
    # A = U * fdiag(S) * VH

    # transform
    A = cf_transform(A, dim_list)

    U, S, VH, stats = fprod.f_svdII(A, gamma, compress_UV=compress_UV, implicit_rank=implicit_rank)
    stats['nrm_A'] = norm(A)
    # return to spatial domain
    if return_spatial:
        U = cf_itransform(U, dim_list)
        S = cf_itransform(reshape(S, (1, *S.shape)), dim_list)[0]  # remove first dimension
        VH = cf_itransform(VH, dim_list)

    return U, S, VH, stats
