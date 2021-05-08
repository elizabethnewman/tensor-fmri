import numpy as np
from numpy.random import randn
from numpy.linalg import norm, qr
from numpy.testing import assert_array_almost_equal, assert_array_equal
from tensor.facewise import t_eye_hat, fdiag, facewise_product
from tensor.utils import reshape, assert_orthogonal, assert_multiple_unitary
import tensor.m_product as mprod
from scipy.linalg import dft
import tensor.t_product as tprod

shape_A = (3, 9, 4, 5, 6)
shape_B = (shape_A[1], 8, *shape_A[2:])

# create orthogonal M
M = []
for i in range(len(shape_A) - 2):
    q, _ = qr(randn(shape_A[i + 2], shape_A[i + 2]))
    M.append(q)


for i in range(len(M)):
    assert_orthogonal(M[i])

A = randn(*shape_A)
B = randn(*shape_B)

# making sure we can multiply
C1 = mprod.m_product(A, B, M)

# force A to be identity tensor for t-product
A = mprod.m_product_eye((shape_A[1], shape_A[1], *shape_A[2:]), M)

C = mprod.m_product(A, B, M)
assert_array_almost_equal(C, B)
err = norm(C - B)
print('mprod: error = %0.2e' % err)

# ==================================================================================================================== #
shape_A = (3, 9, 4, 5, 6)
shape_B = (shape_A[1], 8, *shape_A[2:])

A = randn(*shape_A)
B = randn(*shape_B)

# create identity M
M = []
for i in range(len(shape_A) - 2):
    I = np.eye(shape_A[i + 2])
    M.append(I)


A = randn(*shape_A)
B = randn(*shape_B)

C_true = facewise_product(A, B)
C = mprod.m_product(A, B, M)

assert_array_almost_equal(C, C_true)
err = norm(C - C_true)
print('mprod: error = %0.2e' % err)


# ==================================================================================================================== #
shape_A = (3, 9, 4, 5, 6)
shape_B = (shape_A[1], 8, *shape_A[2:])

# create dct M
M = []
for i in range(len(shape_A) - 2):
    q = (1 / np.sqrt(shape_A[i + 2])) * dft(shape_A[i + 2])
    M.append(q)

for i in range(len(M)):
    assert_multiple_unitary(M[i])

A = randn(*shape_A)
B = randn(*shape_B)

C_true = tprod.t_product(A, B)
# c = prod(shape_A[2:])
# C_true *= c  # renormalize

C = mprod.m_product(A, B, M)

assert_array_almost_equal(C, C_true)
err = norm(C - C_true)
print('mprod with dft: error = %0.2e' % err)


# ==================================================================================================================== #
# check transpose
shape_A = (3, 9, 4, 5, 6)

A = randn(*shape_A)
B = randn(*shape_A)

# create orthogonal M
M = []
for i in range(len(shape_A) - 2):
    q, _ = qr(randn(shape_A[i + 2], shape_A[i + 2]))
    M.append(q)


AT = mprod.m_transpose(A)
ATT = mprod.m_transpose(AT)

assert_array_equal(A, ATT)
err = norm(A - ATT)
print('trans-trans: error = %0.2e' % err)

C1 = mprod.m_product(mprod.m_transpose(A), B, M)
C2 = mprod.m_product(mprod.m_transpose(B), A, M)

assert_array_almost_equal(C1, mprod.m_transpose(C2))
err = norm(C1 - mprod.m_transpose(C2))
print('trans prod.: error = %0.2e' % err)

A_hat = mprod.t_mortho(A, M)
AT_hat = np.swapaxes(A_hat, 0, 1)
AT2 = mprod.t_imortho(AT_hat, M)

assert_array_almost_equal(AT, AT2)
err = norm(AT - AT2)
print('trans test: error = %0.2e' % err)

# ==================================================================================================================== #
shape_A = (10, 15, 3, 12, 10)
A = randn(*shape_A)

# create orthogonal M
M = []
for i in range(len(shape_A) - 2):
    q, _ = qr(randn(shape_A[i + 2], shape_A[i + 2]))
    M.append(q)

U, s, VH = mprod.m_svd(A, M)

A_approx = mprod.m_product(U, mprod.m_product(fdiag(s), VH, M), M)

assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvd: relative error = %0.2e' % err)

for k in range(0, min(A.shape[0], A.shape[1])):
    Uk, sk, VHk = mprod.m_svd(A, M, k + 1)
    Ak = mprod.m_product(Uk, mprod.m_product(fdiag(sk), VHk, M), M)
    err1 = norm(A - Ak) ** 2
    err2 = np.sum(s[k + 1:] ** 2)
    print('%0.2e' % abs(err1 - err2))


# ==================================================================================================================== #
# tsvdII

shape_A = (10, 15, 3, 12, 10)
A = randn(*shape_A)

# create orthogonal M
M = []
for i in range(len(shape_A) - 2):
    q, _ = qr(randn(shape_A[i + 2], shape_A[i + 2]))
    M.append(q)

for i in range(len(M)):
    assert_orthogonal(M[i])

U, s, VH, _ = mprod.m_svdII(A, M, 1)

A_approx = mprod.m_product(U, mprod.m_product(fdiag(s), VH, M), M)

assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvdII: relative error = %0.2e' % err)

nrm_Ahat2 = norm(mprod.t_mortho(A, M)) ** 2
s_hat = np.real(mprod.t_mortho(reshape(s, (1, *s.shape)), M)[0])  # all singular values in transform domain

gamma = np.linspace(0.5, 1, 10)
for k in range(len(gamma)):
    Uk, sk, VHk, multi_rank = mprod.m_svdII(A, M, gamma[k])
    Ak = mprod.m_product(Uk, mprod.m_product(fdiag(sk), VHk, M), M)

    # approximation error
    err1 = norm(A - Ak) ** 2

    # should be equal to the sum of the cutoff singular values in the transform domain
    sk_hat = np.real(mprod.t_mortho(reshape(sk, (1, *sk.shape)), M)[0])
    err2 = np.sum((s_hat - sk_hat) ** 2)  # only count singular values we did not store

    # another approach to get err2
    # sk_hat2 = reshape(s_hat, (s_hat.shape[0], -1))
    # for i in range(min(multi_rank), max(multi_rank) + 1):
    #     sk_hat2[:i, multi_rank >= i] = 0
    # err2 = np.sum(sk_hat2 ** 2)

    nrm_Ak2 = norm(Ak) ** 2
    assert_array_almost_equal(nrm_Ak2, np.sum(sk_hat ** 2))
    gamma_approx = nrm_Ak2 / nrm_Ahat2
    print('err: %0.2e\tgamma diff: %0.2e' % (abs(err1 - err2), (gamma_approx - gamma[k])))

