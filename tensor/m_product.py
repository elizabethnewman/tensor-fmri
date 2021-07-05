import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal
from tensor.utils import assert_compatile_sizes_facewise, assert_compatile_sizes_modek, reshape
import tensor.f_product as fprod
import tensor.mode_k as mode_k


def m_transform(A, M):
    # apply Mk along dimension k
    # each Mk must be orthogonal

    return mode_k.modek_product(A, M, axis=np.arange(2, A.ndim))


def m_itransform(A, M, ortho=True):
    # apply Mk^T along dimension k
    # each Mk^T must be orthogonal

    return mode_k.modek_product(A, M, transpose=ortho, inverse=(not ortho), axis=np.arange(2, A.ndim))


# ==================================================================================================================== #
# m-product
def m_prod(A, B, M, ortho=True):

    assert_compatile_sizes_facewise(A, B)

    if not isinstance(M, tuple) and not isinstance(M, list):
        M = (M,)

    for i in range(len(M)):
        assert_compatile_sizes_modek(A, M[i], i + 2)

    # compute transform
    A_hat = m_transform(A, M)
    B_hat = m_transform(B, M)

    # compute facewise product
    C_hat = fprod.f_prod(A_hat, B_hat)

    # return to spatial comain
    if ortho:
        C = m_itransform(C_hat, M, ortho=ortho)
    else:
        C = m_itransform(C_hat, M, ortho=ortho)

    # ensure C is real-valued
    assert_array_almost_equal(np.imag(C), np.zeros_like(C))
    C = np.real(C)

    return C


def m_eye(shape_I, M, ortho=True):
    assert shape_I[0] == shape_I[1], "Identity tensor must have square frontal slices"

    # create identity tube
    id_tube_hat = np.ones([1, 1, *shape_I[2:]])
    id_tube = m_itransform(id_tube_hat, M, ortho=ortho)

    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = id_tube

    return I


def m_tran(A):

    return np.swapaxes(A, 0, 1)


# ==================================================================================================================== #
def m_svd(A, M, k=None, ortho=True):
    # A = U * fdiag(S) * VH

    shape_A = A.shape

    # transform
    A = m_transform(A, M)

    U, s, VH, stats = fprod.f_svd(A, k)

    # return to spatial domain
    U = m_itransform(U, M, ortho=ortho)
    S = np.reshape(m_itransform(reshape(s, (1, *s.shape)), M, ortho=ortho), (s.shape[0], *shape_A[2:]))  # remove first dimension
    VH = m_itransform(VH, M, ortho=ortho)

    # store statistics
    stats['nrm_A'] = norm(A)
    stats['M_storage'] = m_storage(M)
    stats['svd_storage'] = stats['total_storage']
    stats['total_storage'] = stats['svd_storage'] + stats['M_storage']
    stats['compression_ratio'] = A.size / stats['total_storage']

    return U, S, VH, stats


def m_svdII(A, M, gamma, compress_UV=True, return_spatial=True, implicit_rank=None, ortho=True):
    # A = U * fdiag(S) * VH

    # transform
    A = m_transform(A, M)
    # nrm_Ahat = np.linalg.norm(A)

    U, S, VH, stats = fprod.f_svdII(A, gamma, compress_UV=compress_UV, implicit_rank=implicit_rank)

    # return to spatial domain
    if return_spatial:
        U = m_itransform(U, M, ortho=ortho)
        S = m_itransform(np.reshape(S, (1, *S.shape)), M, ortho=ortho)[0]   # remove first dimension
        VH = m_itransform(VH, M, ortho=ortho)

    # store statistics
    stats['nrm_A'] = norm(A)
    stats['M_storage'] = m_storage(M)
    stats['svd_storage'] = stats['total_storage']
    stats['total_storage'] = stats['svd_storage'] + stats['M_storage']
    stats['compression_ratio'] = A.size / stats['total_storage']

    return U, S, VH, stats


def m_storage(M):
    if isinstance(M, tuple) or isinstance(M, list):
        n = 0
        for i in range(len(M)):
            n += M[i].size
    else:
        n = M.size

    return n
