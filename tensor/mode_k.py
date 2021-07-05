import numpy as np
from tensor.utils import assert_compatile_sizes_modek, reshape, make_axis_iterable
from numpy.linalg import inv


# mode-k product and unfolding
def modek_product_one_dimension(A, M, k, transpose=False, inverse=False):
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
    if transpose or inverse:
        if inverse:
            A_hat = np.moveaxis(np.linalg.solve(M, np.moveaxis(A, k, -2)), -2, -1)
        else:
            A_hat = np.moveaxis(A, k, -1) @ np.conjugate(M)
    else:
        A_hat = np.moveaxis(A, k, -1) @ M.T

    A_hat = np.moveaxis(A_hat, -1, k)

    return A_hat


def modek_product(A, M, axis=None, transpose=False, inverse=False):
    # apply matrix or tuple or list of matrices M along various axes
    # axis is an int or list of ints along which to apply each M
    # order of matrices M must match the order of axis

    # ensure M is iterable
    if not isinstance(M, tuple) and not isinstance(M, list):
        M = (M,)

    if axis is None:
        # default: apply to first dimensions
        axis = np.arange(len(M))
    else:
        axis = make_axis_iterable(axis)

    for i in range(len(axis)):
        if M[i] is not None:
            A = modek_product_one_dimension(A, M[i], axis[i], transpose=transpose, inverse=inverse)

    return A


def modek_unfold(A, k):
    # A is a tensor
    A = np.moveaxis(A, k, 0)
    return reshape(A, (A.shape[0], -1))


def modek_fold(A, k, shape_A):
    # A is a matrix
    # shape_A is the shape of A before unfolding, as a tensor

    if isinstance(shape_A, list):
        shape_A = tuple(shape_A)

    A = reshape(A, (-1,) + shape_A[:k] + shape_A[k+1:])

    return np.moveaxis(A, 0, k)

