import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.facewise as fprod
from tensor.utils import reshape
import time

# ==================================================================================================================== #
# facewise diagonal
rank = 3
shape_D = (rank, rank, 4, 5, 6)
D_true = np.zeros(shape_D)
d = np.zeros((rank,) + shape_D[2:])
for i in range(shape_D[2]):
    for j in range(shape_D[3]):
        for k in range(shape_D[4]):
            d_tmp = randn(rank)
            d[:, i, j, k] = d_tmp
            D_true[:, :, i, j, k] = np.diag(d_tmp)

D_hat = fprod.fdiag(d)

assert_array_equal(D_hat, D_true)
err = norm(D_hat - D_true) / norm(D_true)
print('fdiag: relative error = %0.2e' % err)

# ==================================================================================================================== #
# facewise
shape_A = (2, 3, 4, 5, 6)
shape_B = (shape_A[1], 8, *shape_A[2:])
shape_C = (shape_A[0], shape_B[1], *shape_A[2:])

A = randn(*shape_A)
B = randn(*shape_B)

C_true = np.zeros(shape_C)
for i in range(shape_A[2]):
    for j in range(shape_A[3]):
        for k in range(shape_A[4]):
            tmp_A = A[:, :, i, j, k]
            tmp_B = B[:, :, i, j, k]
            C_true[:, :, i, j, k] = tmp_A @ tmp_B

C = fprod.facewise_product(A, B)

assert_array_almost_equal(C, C_true)
err = norm(C - C_true) / norm(C_true)

print('facewise: relative error = %0.2e' % err)

# ==================================================================================================================== #
# facewise svd
shape_A = (10, 15, 3, 12, 10)
A = randn(*shape_A)

U, s, VH = fprod.facewise_t_svd(A)

A_approx = fprod.facewise_product(U, fprod.facewise_product(fprod.fdiag(s), VH))

assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvd: relative error = %0.2e' % err)

for k in range(0, min(A.shape[0], A.shape[1])):
    Uk, sk, VHk = fprod.facewise_t_svd(A, k + 1)
    Ak = fprod.facewise_product(Uk, fprod.facewise_product(fprod.fdiag(sk), VHk))
    err1 = norm(A - Ak) ** 2
    err2 = np.sum(s[k + 1:] ** 2)
    print('%0.2e' % abs(err1 - err2))

# ==================================================================================================================== #
# test facewise fast

k = 5

for n in (10, 50, 100, 500):
    shape_A = (n, n, 1000)
    A = randn(*shape_A)

    start1 = time.time()
    u1, s1, vh1 = fprod.facewise_t_svd(A, k)
    end1 = time.time()
    print('time1 = %0.4e' % (end1 - start1))

    start2 = time.time()
    u2, s2, vh2 = fprod.facewise_t_svd_fast(A, k)
    end2 = time.time()
    print('time2 = %0.4e' % (end2 - start2))

    assert_array_almost_equal(u1, u2)
    assert_array_almost_equal(s1, s2)
    assert_array_almost_equal(vh1, vh2)



# ==================================================================================================================== #
# facewise tsvdII

eps = np.finfo(np.float64).eps

shape_A = (10, 15, 3, 12, 10)
A = randn(*shape_A)

U, s, VH, _ = fprod.facewise_t_svdII(A, 1)

A_approx = fprod.facewise_product(U, fprod.facewise_product(fprod.fdiag(s), VH))

# assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvdII: relative error = %0.2e' % err)

nrm_A2 = norm(A) ** 2
nrm_Ahat = norm(A) ** 2
gamma = np.linspace(0.5, 1, 10)
for k in range(len(gamma)):
    Uk, sk, VHk, multi_rank = fprod.facewise_t_svdII(A, gamma[k])
    Ak = fprod.facewise_product(Uk, fprod.facewise_product(fprod.fdiag(sk), VHk))

    # approximation error
    err1 = norm(A - Ak) ** 2

    # should be equal to the sum of the cutoff singular values in the transform domain
    err2 = np.sum((s - sk) ** 2)

    # approximation of energy
    gamma_approx = (norm(Ak) / nrm_Ahat) ** 2

    print('err: %0.2e\tgamma diff: %0.2e' % (abs(err1 - err2), (gamma_approx - gamma[k])))
