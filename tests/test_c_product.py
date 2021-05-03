import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.utils as tu
import tensor.c_product as cprod


# ==================================================================================================================== #
# identity tensors
I_hat = tu.t_eye_hat((3, 3, 4, 5, 6))

cprod_I_true = cprod.t_idct(I_hat)
cprod_I = cprod.c_product_eye(I_hat.shape)

assert_array_almost_equal(cprod_I, cprod_I_true)
err = norm(cprod_I - cprod_I_true)
print('cprod eye: error = %0.2e' % err)


shape_A = (3, 9, 4, 5, 6)
shape_B = (shape_A[1], 8, *shape_A[2:])

A = randn(*shape_A)
B = randn(*shape_B)

# making sure we can multiply
C1 = cprod.c_product(A, B)

# force A to be identity tensor for t-product
A = cprod.c_product_eye((shape_A[1], shape_A[1], *shape_A[2:]))

C = cprod.c_product(A, B)
assert_array_almost_equal(C, B)
print('cprod: error = %0.2e' % err)

# ==================================================================================================================== #
# check transpose
shape_A = (3, 9, 4, 5, 6)

A = randn(*shape_A)
B = randn(*shape_A)

AT = cprod.c_transpose(A)
ATT = cprod.c_transpose(AT)

assert_array_equal(A, ATT)
err = norm(A - ATT)
print('trans-trans: error = %0.2e' % err)

C1 = cprod.c_product(cprod.c_transpose(A), B)
C2 = cprod.c_product(cprod.c_transpose(B), A)

assert_array_almost_equal(C1, cprod.c_transpose(C2))
err = norm(C1 - cprod.c_transpose(C2))
print('trans prod.: error = %0.2e' % err)

A_hat = cprod.t_dct(A)
AT_hat = np.swapaxes(A_hat, 0, 1)
AT2 = cprod.t_idct(AT_hat)

assert_array_almost_equal(AT, AT2)
err = norm(AT - AT2)
print('trans test: error = %0.2e' % err)

# ==================================================================================================================== #
shape_A = (15, 10, 3, 12, 10)
A = randn(*shape_A)

U, s, VH = cprod.c_svd(A)

A_approx = cprod.c_product(U, cprod.c_product(tu.fdiag(s), VH))

assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvd: relative error = %0.2e' % err)

for k in range(0, min(A.shape[0], A.shape[1])):
    Uk, sk, VHk = cprod.c_svd(A, k + 1)
    Ak = cprod.c_product(Uk, cprod.c_product(tu.fdiag(sk), VHk))
    err1 = norm(A - Ak) ** 2
    err2 = np.sum(s[k + 1:] ** 2)
    print('%0.2e' % abs(err1 - err2))


# ==================================================================================================================== #
# facewise c_svdII

eps = np.finfo(np.float64).eps

shape_A = (15, 10, 3, 10)
A = randn(*shape_A)

U, s, VH, _ = cprod.c_svdII(A, 1)

A_approx = cprod.c_product(U, cprod.c_product(tu.fdiag(s), VH))

assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvdII: relative error = %0.2e' % err)


nrm_Ahat = norm(cprod.t_dct(A))

# nrm_A2 = norm(A) ** 2
gamma = np.linspace(0.5, 1, 10)
for k in range(len(gamma)):
    Uk, sk, VHk, multi_rank = cprod.c_svdII(A, gamma[k])
    Ak = cprod.c_product(Uk, cprod.c_product(tu.fdiag(sk), VHk))

    sk_hat = cprod.t_dct(np.reshape(s, (1, *s.shape)))[0]
    sk_hat = np.reshape(sk_hat, (sk_hat.shape[0], -1))
    for i in range(min(multi_rank), max(multi_rank) + 1):
        sk_hat[:i, multi_rank >= i] = 0

    err1 = norm(A - Ak) ** 2
    err2 = np.sum(sk_hat ** 2)

    gamma_approx = (norm(Ak) / nrm_Ahat) ** 2
    print('err: %0.2e\tgamma: %0.2e' % (abs(err1 - err2), (gamma_approx - gamma[k])))

