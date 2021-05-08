import numpy as np
from numpy.testing import assert_array_almost_equal
from tensor.utils import assert_compatile_sizes_facewise, assert_compatile_sizes_modek, reshape
import tensor.facewise as fprod
import tensor.mode_k as mode_k


def t_mortho(A, M):
    # apply Mk along dimension k
    # each Mk must be orthogonal

    for i in range(len(M)):
        A = mode_k.modek_product(A, M[i], i + 2)

    return A


def t_imortho(A, M):
    # apply Mk^T along dimension k
    # each Mk^T must be orthogonal

    for i in range(len(M) - 1, -1, -1):
        if np.all(np.isreal(M[i])):
            A = mode_k.modek_product(A, M[i].T, i + 2)
        else:
            A = mode_k.modek_product(A, np.conjugate(M[i]).T, i + 2)

    return A



# ==================================================================================================================== #
# m-product
def m_product(A, B, M, ortho=True):

    assert_compatile_sizes_facewise(A, B)

    for i in range(len(M)):
        assert_compatile_sizes_modek(A, M[i], i + 2)

    if ortho:
        # move to transform domain
        A_hat = t_mortho(A, M)
        B_hat = t_mortho(B, M)
    else:
        raise ValueError("m-product for non-orthogonal matrices not yet implemented")

    # compute facewise product
    C_hat = fprod.facewise_product(A_hat, B_hat)

    # return to spatial comain
    C = t_imortho(C_hat, M)

    # ensure C is real-valued
    assert_array_almost_equal(np.imag(C), np.zeros_like(C))
    C = np.real(C)

    return C


def m_product_eye(shape_I, M):
    assert shape_I[0] == shape_I[1], "Identity tensor must have square frontal slices"

    # create identity tube
    id_tube_hat = np.ones([1, 1, *shape_I[2:]])
    id_tube = t_imortho(id_tube_hat, M)

    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = id_tube

    return I


def m_transpose(A):

    return np.swapaxes(A, 0, 1)


# ==================================================================================================================== #
def m_svd(A, M, k=None):
    # A = U * fdiag(S) * VH

    shape_A = A.shape

    # transform
    A = t_mortho(A, M)

    U, s, VH = fprod.facewise_t_svd(A, k)

    # return to spatial domain
    U = t_imortho(U, M)
    S = np.reshape(t_imortho(reshape(s, (1, *s.shape)), M), (s.shape[0], *shape_A[2:]))  # remove first dimension
    VH = t_imortho(VH, M)

    return U, S, VH


def m_svdII(A, M, gamma, compress_UV=True, return_spatial=True):
    # A = U * fdiag(S) * VH

    # transform
    A = t_mortho(A, M)
    # nrm_Ahat = np.linalg.norm(A)

    U, S, VH, multi_rank = fprod.facewise_t_svdII(A, gamma, compress_UV=compress_UV)

    # return to spatial domain
    if return_spatial:
        U = t_imortho(U, M)
        S = t_imortho(np.reshape(S, (1, *S.shape)), M)[0]   # remove first dimension
        VH = t_imortho(VH, M)

    return U, S, VH, multi_rank