import numpy as np
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
# make iterable
def make_axis_iterable(A):
    if not isinstance(A, tuple) and not isinstance(A, list) and not isinstance(A, np.ndarray):
        A = (A,)
    return A


# ==================================================================================================================== #
# reshaping commands
def reshape(A, newshape):
    # column-wise vectorization
    return np.reshape(A, newshape, 'F')


def unfold(A, dim=1):
    # unfold where dim becomes the columns of a matrix
    A = np.moveaxis(A, dim, -1)
    return reshape(A, (-1, A.shape[-1]))


def fold(A, shape_A, dim=1):
    # A is an unfolded tensor
    # shape_A is the desired shape
    shape_A = tuple(shape_A)
    A = reshape(A, shape_A[:dim] + shape_A[dim + 1:] + (-1,))
    return np.moveaxis(A, -1, dim)

# ==================================================================================================================== #
# facewise diagonal
def f_diag(d):
    # d is a tensor of size k x n2 x ... x nd
    # Turn d into a facewise diagonal tensor of size k x k x n2 x ... x nd
    D = np.zeros([d.shape[0], d.shape[0], *d.shape[1:]])
    idx = np.arange(d.shape[0])
    D[idx, idx] = d

    return D

