import numpy as np
from numpy.linalg import svd
from numpy.testing import assert_array_almost_equal
from tensor.utils import assert_compatile_sizes_facewise

# ==================================================================================================================== #
# facewise
def facewise_product(A, B):
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


# ==================================================================================================================== #
def facewise_t_svd(A, k=None):
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

    return u, s, vh


def facewise_t_svdII(A, gamma, compress_UV=False):

    # compute full t-svd
    U, s, VH = facewise_t_svd(A)

    if gamma is not None and gamma < 1:
        d2 = np.flip(np.sort(s.reshape(-1) ** 2))

        idx = np.argwhere(np.cumsum(d2) / np.sum(d2) < gamma)[-1]

        cutoff = np.sqrt(d2[idx])

        s[s < cutoff] = 0

    multi_rank = np.sum(s.reshape(s.shape[0], -1) != 0, axis=0)

    if gamma is not None and gamma < 1 and compress_UV:
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

