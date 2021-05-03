import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.utils as tu
import tensor.t_product as tprod
from math import prod


np.random.seed(20)
# ==================================================================================================================== #
# identity tensors
I_hat = tu.t_eye_hat((3, 3, 4, 5, 6))

tprod_I_true = tprod.t_ifft(I_hat)
tprod_I = tprod.t_product_eye(I_hat.shape)

assert_array_almost_equal(tprod_I, tprod_I_true)
err = norm(tprod_I - tprod_I_true)
print('eye: error = %0.2e' % err)


shape_A = (3, 9, 4, 5, 6)
shape_B = (shape_A[1], 8, *shape_A[2:])

A = randn(*shape_A)
B = randn(*shape_B)

# making sure we can multiply
C1 = tprod.t_product(A, B)

# force A to be identity tensor for t-product
A = tprod.t_product_eye((shape_A[1], shape_A[1], *shape_A[2:]))

C = tprod.t_product(A, B)
assert_array_almost_equal(C, B)
err = norm(C - B)
print('tprod: error = %0.2e' % err)

# ==================================================================================================================== #
# check transpose
shape_A = (3, 4, 5, 6, 7)

A = randn(*shape_A)
B = randn(*shape_A)

AT = tprod.t_transpose(A)
ATT = tprod.t_transpose(AT)

assert_array_equal(A, ATT)
err = norm(A - ATT)
print('trans-trans: error = %0.2e' % err)

C1 = tprod.t_product(tprod.t_transpose(A), B)
C2 = tprod.t_product(tprod.t_transpose(B), A)

assert_array_almost_equal(C1, tprod.t_transpose(C2))
err = norm(C1 - tprod.t_transpose(C2))
print('trans prod.: error = %0.2e' % err)

A_hat = tprod.t_fft(A)
AT_hat = np.conjugate(np.swapaxes(A_hat, 0, 1))
AT2 = tprod.t_ifft(AT_hat)

assert_array_almost_equal(AT, AT2)
err = norm(AT - AT2)
print('trans test: error = %0.2e' % err)


# ==================================================================================================================== #
shape_A = (10, 15, 3, 12, 10)
A = randn(*shape_A)

U, s, VH = tprod.t_svd(A)

A_approx = tprod.t_product(U, tprod.t_product(tu.fdiag(s), VH))

assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvd: relative error = %0.2e' % err)

for k in range(0, min(A.shape[0], A.shape[1])):
    Uk, sk, VHk = tprod.t_svd(A, k + 1)
    Ak = tprod.t_product(Uk, tprod.t_product(tu.fdiag(sk), VHk))
    err1 = norm(A - Ak) ** 2
    err2 = np.sum(s[k + 1:] ** 2)
    print('%0.2e' % abs(err1 - err2))


# ==================================================================================================================== #
# facewise tsvdII

shape_A = (10, 15, 4)
A = randn(*shape_A)

U, s, VH, _ = tprod.t_svdII(A, 1)

A_approx = tprod.t_product(U, tprod.t_product(tu.fdiag(s), VH))

# assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('tsvdII: relative error = %0.2e' % err)

nrm_Ahat = norm(tprod.t_fft(A))
nrm_A = norm(A)

# transformation is a multiple of a unitary matrix (this is the default)
c = prod(shape_A[2:])

gamma = np.linspace(0.5, 1, 10)
for k in range(len(gamma)):
    Uk, sk, VHk, multi_rank = tprod.t_svdII(A, gamma[k])
    Ak = tprod.t_product(Uk, tprod.t_product(tu.fdiag(sk), VHk))

    sk_hat = np.real(tprod.t_fft(np.reshape(s, (1, *s.shape)))[0])
    sk_hat = np.reshape(sk_hat, (sk_hat.shape[0], -1))
    for i in range(min(multi_rank), max(multi_rank) + 1):
        sk_hat[:i, multi_rank >= i] = 0

    # need to rescale because we do not use the orthogonal fft
    err1 = norm(A - Ak) ** 2
    err2 = np.sum(sk_hat ** 2)

    gamma_approx = (norm(Ak) / nrm_Ahat) ** 2
    print('err: %0.2e\tgamma: %0.2e' % (abs(err1 - err2), (gamma_approx - gamma[k])))


