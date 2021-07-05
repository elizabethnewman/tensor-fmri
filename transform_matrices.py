import numpy as np
from tensor.mode_k import modek_unfold
from numpy.linalg import eig
import math


def build_haar_matrix(n, normalized=True):
    # n is the power of 2
    if not (2 ** np.log2(n) == n):
        raise ValueError('haar_matrix: only available for powers of 2')

    if n > 2:
        M = build_haar_matrix(n // 2)
    else:
        return np.array([[1, 1], [1, -1]])

    M_n = np.kron(M, [1, 1])
    M_i = np.kron(np.eye(len(M)), [1, -1])
    if normalized:
        M_i *= math.sqrt(n // 2)

    M = np.vstack((M_n, M_i))
    return M


def haar_matrix(n, normalized=True):
    M = build_haar_matrix(n, normalized=normalized)
    if normalized:
        M = M / math.sqrt(n)
    return M


def banded_matrix(n, band):
    """

    Parameters
    ----------
    n : size of matrix (n x n)
    band : bandwidth (band=0 -> diagonal matrix, band=n-1 -> dense matrix)

    Returns
    -------

    """
    c = np.stack((np.ones(band), np.zeros(n - band)))
    r = np.stack((np.ones(1), np.zeros(n - 1)))
    M = np.linalg.toeplitz(r, c)

    # normalize
    M /= np.sum(M, axis=0)

    return M


def random_orthogonal_matrix(n):
    q, _ = np.linalg.qr(np.random.randn(n, n))
    return q


def left_singular_matrix(A, k):
    Ak = modek_unfold(A, k)
    d, u = eig(Ak @ Ak.T)

    idx = np.argsort(d)
    idx = np.flip(idx)

    return u[:, idx]
