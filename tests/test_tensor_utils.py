import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.utils as tu


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

C = tu.facewise(A, B)

assert_array_almost_equal(C, C_true)
err = norm(C - C_true) / norm(C_true)

print('facewise: relative error = %0.2e' % err)

# ==================================================================================================================== #
# modek
k = 4
A = randn(*shape_A)
M = randn(10, A.shape[k])

A_hat_true = np.swapaxes(A, 0, k)
shape_A_tmp = A_hat_true.shape

A_hat_true = np.reshape(A_hat_true, (A.shape[k], -1))
A_hat_true = M @ A_hat_true
A_hat_true = np.reshape(A_hat_true, (M.shape[0], *shape_A_tmp[1:]))
A_hat_true = np.swapaxes(A_hat_true, 0, k)

A_hat = tu.modek_product(A, M, k)

assert_array_almost_equal(A_hat, A_hat_true)
err = norm(A_hat - A_hat_true) / norm(A_hat_true)
print('modek: relative error = %0.2e' % err)

k = 3
A = randn(*shape_A)
M = np.ones((1, A.shape[k]))

A_hat_true = np.sum(A, axis=k, keepdims=True)

A_hat = tu.modek_product(A, M, k)

assert_array_almost_equal(A_hat, A_hat_true)
err = norm(A_hat - A_hat_true) / norm(A_hat_true)
print('modek sum: relative error = %0.2e' % err)


# ==================================================================================================================== #
# facewise diagonal
d = randn(3, 4, 5, 6)
D = tu.fdiag(d)

# ==================================================================================================================== #
# fold and unfold
shape_A = (2, 3, 1, 4, 5)
A = randn(*shape_A)

k = 3

A_mat = tu.modek_unfold(A, k)
A_ten = tu.modek_fold(A_mat, k, A.shape)

assert_array_equal(A_ten, A)
err = norm(A_ten - A) / norm(A)
print('modek fold/unfold: relative error = %0.2e' % err)

# random selection
assert_array_equal(A_mat[:, 0], A[0, 0, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 1], A[0, 0, 0, :, 1].reshape(-1))
assert_array_equal(A_mat[:, 2], A[0, 0, 0, :, 2].reshape(-1))
assert_array_equal(A_mat[:, 5], A[1, 0, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 6], A[1, 0, 0, :, 1].reshape(-1))
assert_array_equal(A_mat[:, 10], A[0, 1, 0, :, 0].reshape(-1))

# ==================================================================================================================== #
# facewise svd
shape_A = (10, 15, 3, 12, 10)
A = randn(*shape_A)

U, s, VH = tu.facewise_t_svd(A)

A_approx = tu.facewise(U, tu.facewise(tu.fdiag(s), VH))

assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvd: relative error = %0.2e' % err)

for k in range(0, min(A.shape[0], A.shape[1])):
    Uk, sk, VHk = tu.facewise_t_svd(A, k + 1)
    Ak = tu.facewise(Uk, tu.facewise(tu.fdiag(sk), VHk))
    err1 = norm(A - Ak) ** 2
    err2 = np.sum(s[k + 1:] ** 2)
    print('%0.2e' % abs(err1 - err2))


# ==================================================================================================================== #
# facewise tsvdII

eps = np.finfo(np.float64).eps

shape_A = (10, 15, 3, 12, 10)
A = randn(*shape_A)

U, s, VH, _ = tu.facewise_t_svdII(A, 1)

A_approx = tu.facewise(U, tu.facewise(tu.fdiag(s), VH))

# assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvdII: relative error = %0.2e' % err)

nrm_A2 = norm(A) ** 2
gamma = np.linspace(0.5, 1, 10)
for k in range(len(gamma)):
    Uk, sk, VHk, multi_rank = tu.facewise_t_svdII(A, gamma[k])
    Ak = tu.facewise(Uk, tu.facewise(tu.fdiag(sk), VHk))

    err1 = norm(A - Ak) ** 2
    err2 = np.sum((s * (sk == 0)) ** 2)

    print('err: %0.2e' % abs(err1 - err2))



