import numpy as np
from numpy.linalg import svd
from numpy.testing import assert_array_almost_equal



# ==================================================================================================================== #
# compatibility assertions
def assert_compatile_sizes_facewise(A, B):
    """
    Determines if two tensors can be multiplied together facewise
    """
    shape_A = A.shape
    shape_B = B.shape

    assert len(shape_A) == len(shape_B), "Tensors need to be of the same order"
    assert shape_A[1] == shape_B[0], "First two dimensions of tensors must be compatible"
    assert shape_A[2:] == shape_B[2:], "Tensors must have same third through d dimensions"


def assert_compatile_sizes_modek(A, M, k):
    """
    Determines if a matrix or tuple/list of matrices can be applied along mode k
    """
    assert M.shape[1] == A.shape[k], "Matrix must be of compatible size with tensor dimension"


def assert_orthogonal(M):
    assert M.shape[0] == M.shape[1], "Orthogonal matrices must be square"

    I = np.eye(M.shape[0])
    assert_array_almost_equal(M.T @ M, I)
    assert_array_almost_equal(M @ M.T, I)


def assert_multiple_unitary(M):
    assert M.shape[0] == M.shape[1], "Orthogonal matrices must be square"

    MTM = np.conjugate(M).T @ M
    d = np.diag(MTM)
    d1 = d / d[0]
    e = np.ones(d.shape)

    MTMd = MTM - np.diag(d)
    Z = np.zeros(M.shape)

    assert_array_almost_equal(MTMd, Z)
    assert_array_almost_equal(d1, e)


# ==================================================================================================================== #
# facewise
def facewise(A, B):
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
# mode-k product and unfolding
def modek_product(A, M, k, transpose=False):
    """

    Parameters
    ----------
    A : (n0 x n1 x ... x n_{k-1} x nk x n_{k+1} x ... x nd) array
    M : p x nk array
    k : dimension along which to apply M, int between 0 and d

    Returns
    -------
    A_hat : (n0 x n1 x ... x n_{k-1} x p x n_{k+1} x ... x nd) array
    """

    assert_compatile_sizes_modek(A, M, k)

    # apply M to tubes of A (note that the transpose is reversed because we apply M on the right)
    if transpose:
        A_hat = np.swapaxes(A, k, -1) @ M
    else:
        A_hat = np.swapaxes(A, k, -1) @ M.T

    return np.swapaxes(A_hat, k, -1)


def modek_product_many(A, M, axis=None, transpose=False):
    # apply tuple or list of matrices M along various axes
    # axis is an int or list of ints along which to apply each M
    # M must be ordered according to dim

    if not isinstance(M, tuple) and not isinstance(M, list):

        if axis is not None:
            if isinstance(axis, tuple) or isinstance(axis, list):
                # apply the same matrix M along many axes
                for i in range(len(axis)):
                    A = modek_product(A, M, axis[i], transpose=transpose)
            else:
                A = modek_product(A, M, axis, transpose=transpose)

        else:
            # default to apply to first dimension of A
            ValueError("Must provide axis along which to apply matrix")

    else:
        if axis is None:
            # default: apply to first dimensions
            axis = np.arange(len(M))

        if isinstance(axis, int):
            axis = (axis,)

        for i in range(len(axis)):
            A = modek_product(A, M[i], axis[i], transpose=transpose)

    return A




def modek_unfold(A, k):
    # A is a tensor
    A = np.swapaxes(A, 0, k)
    return np.reshape(A, (A.shape[0], -1))


def modek_fold(A, k, shape_A):
    # A is a matrix
    # shape_A is the shape of A before unfolding, as a tensor

    if isinstance(shape_A, tuple):
        shape_A = list(shape_A)

    shape_A[k] = shape_A[0]
    shape_A = shape_A[1:]
    A = np.reshape(A, (A.shape[0], *shape_A))

    return np.swapaxes(A, 0, k)


# ==================================================================================================================== #
# general purpose functions
def t_eye_hat(shape_I):
    # identity tensor is the same in the transform domain
    I = np.zeros(shape_I)
    idx = np.arange(shape_I[0])
    I[idx, idx] = 1.0

    return I


def fdiag(d):
    # d is a tensor of size k x n2 x ... x nd
    # Turn d into a facewise diagonal tensor of size k x k x n2 x ... x nd
    D = np.zeros([d.shape[0], d.shape[0], *d.shape[1:]])
    idx = np.arange(d.shape[0])
    D[idx, idx] = d

    return D

# add frobenius norm?  that's the default in numpy

# ==================================================================================================================== #

def facewise_t_svd(A, k=None):
    # A = U * fdiag(S) * VH

    shape_A = A.shape

    # transform into third-order tensor
    A = np.reshape(A, (A.shape[0], A.shape[1], -1))

    # determine sizes
    if k is None:
        k = min(A.shape[0], A.shape[1])
    else:
        k = min(k, min(A.shape[0], A.shape[1]))

    # pre-allocate
    U = np.zeros((A.shape[0], k, A.shape[2]), dtype=A.dtype)
    S = np.zeros((k, A.shape[2]), dtype=np.float64)
    VH = np.zeros((k, A.shape[1], A.shape[2]), dtype=A.dtype)

    # form SVD
    for i in range(A.shape[2]):
        u, s, vh = svd(A[:, :, i], full_matrices=False)
        U[:, :, i] = u[:, :k]
        S[:, i] = s[:k]
        VH[:, :, i] = vh[:k, :]

    # reshape
    U = np.reshape(U, (U.shape[0], U.shape[1], *shape_A[2:]))
    S = np.reshape(S, (S.shape[0], *shape_A[2:]))
    VH = np.reshape(VH, (VH.shape[0], VH.shape[1], *shape_A[2:]))

    return U, S, VH


def facewise_t_svdII(A, gamma, compress_UV=False):

    # compute full t-svd
    U, s, VH = facewise_t_svd(A)

    if gamma is not None and gamma < 1:
        d2 = np.flip(np.sort(s.reshape(-1) ** 2))

        idx = np.argwhere(np.cumsum(d2) / np.sum(d2) < gamma)[-1]

        cutoff = np.sqrt(d2[idx])

        s[s < cutoff] = 0

    multi_rank = np.sum(s.reshape(s.shape[0], -1) != 0, axis=0)

    if compress_UV:
        # zero out columns/rows corresponding to zero singular values
        shape_U = U.shape
        shape_VH = VH.shape

        U = np.reshape(U, (U.shape[0], U.shape[1], -1))
        VH = np.reshape(VH, (VH.shape[0], VH.shape[1], -1))

        for i in range(min(multi_rank), max(multi_rank)):
            U[:, i:, multi_rank <= i] = 0
            VH[i:, :, multi_rank <= i] = 0

        U = np.reshape(U, shape_U)
        VH = np.reshape(VH, shape_VH)

    return U, s, VH, multi_rank
