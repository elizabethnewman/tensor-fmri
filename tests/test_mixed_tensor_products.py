import numpy as np
from numpy.random import randn
from numpy.linalg import norm, qr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from tensor.utils import reshape, f_diag
import tensor.tensor_product_wrapper as tp

np.random.seed(20)
# ==================================================================================================================== #
# choose product type {'cf', 'tf'}
prod_type = 'cf'

dim_list=()
str_=str(input("Input dimensions: "))
list_=str_.split(" ")
for i in list_:
    dim_list += (int(i),)


# ==================================================================================================================== #
# helper functions
def create_m(shape_in, shape_out):
    assert len(shape_in) == len(shape_out)

    M = []
    for i in range(len(shape_in)):
        M.append(randn(shape_out[i], shape_out[i]))
    return M


def create_orthogonal_m(shape):
    M = []
    for i in range(len(shape)):
        q, _ = qr(randn(shape[i], shape[i]))
        M.append(q)
    return M

# ==================================================================================================================== #
# identity tensors
shape_I = (3, 3, 4, 5, 6)

M = create_orthogonal_m(shape_I[2:])

I_hat_true = np.zeros(shape_I)

I_hat_true = reshape(I_hat_true, shape_I[0:2] + (-1,))
for i in range(I_hat_true.shape[2]):
    I_hat_true[:, :, i] = np.eye(shape_I[0])

I_hat_true = reshape(I_hat_true, shape_I)

I_true = tp.ten_itransform(I_hat_true, dim_list, prod_type=prod_type, M=M)
I_approx = tp.ten_eye(shape_I, dim_list, prod_type=prod_type, M=M)

assert_array_almost_equal(I_approx, I_true)
err = norm(I_approx - I_true)
print('identity error: error = %0.2e' % err)

# ==================================================================================================================== #
# check basic multiplication with identity matrices
shape_A = (3, 3, 4, 5, 6)
shape_B = (shape_A[1], 8, *shape_A[2:])

A = tp.ten_eye(shape_A, dim_list)
B = randn(*shape_B)

C = tp.ten_prod(A, B, dim_list)
assert_array_almost_equal(C, B)
print('left identity multiply error = %0.2e' % err)

shape_A = (8, 3, 4, 5, 6)
shape_B = (shape_A[1], shape_A[1], *shape_A[2:])

A = randn(*shape_A)
B = tp.ten_eye(shape_B, dim_list)

C = tp.ten_prod(A, B, dim_list)
assert_array_almost_equal(C, A)
print('right identity multiply error = %0.2e' % err)

# ==================================================================================================================== #
# transpose
shape_A = (3, 9, 4, 5, 6)
M = create_orthogonal_m(shape_A[2:])
A = randn(*shape_A)

AT = tp.ten_tran(A, prod_type=prod_type)
ATT = tp.ten_tran(AT, prod_type=prod_type)
assert_array_equal(A, ATT)

A_hat = tp.ten_transform(A, dim_list, prod_type=prod_type, M=M)
AT_hat = np.swapaxes(np.conjugate(A_hat), 0, 1)
AT_approx = tp.ten_itransform(AT_hat, dim_list, prod_type=prod_type, M=M)

assert_array_almost_equal(AT, AT_approx)
err = norm(AT - AT_approx)
print('transpose error = %0.2e' % err)
# ==================================================================================================================== #
# check transpose multiplication
shape_A = (3, 9, 4, 5, 6)

M = create_orthogonal_m(shape_A[2:])

A = randn(*shape_A)
B = randn(*shape_A)

C1 = tp.ten_prod(tp.ten_tran(A, prod_type=prod_type), B, dim_list, prod_type=prod_type, M=M)
C2 = tp.ten_tran(tp.ten_prod(tp.ten_tran(B, prod_type=prod_type), A, dim_list, prod_type=prod_type, M=M), prod_type=prod_type)

assert_array_almost_equal(C1, C2)
err = norm(C1 - C2)
print('product with transposes error = %0.2e' % err)

# ==================================================================================================================== #
# check tsvd
shape_A = (15, 10, 3, 12, 10)
M = create_orthogonal_m(shape_A[2:])
A = randn(*shape_A)

u, s, vh, stats = tp.ten_svd(A, dim_list, prod_type=prod_type, M=M)

A_approx = tp.ten_prod(u, tp.ten_prod(f_diag(s), vh, dim_list, prod_type=prod_type, M=M), dim_list, prod_type=prod_type, M=M)

assert_array_almost_equal(A, A_approx)
rel_err = norm(A - A_approx) / norm(A)
print('full tsvd relative error = %0.2e' % rel_err)

print('truncated tsvd:')
for k in range(0, min(A.shape[0], A.shape[1]) - 1):
    uk, sk, vhk, statsk = tp.ten_svd(A, dim_list, k=k + 1, prod_type=prod_type, M=M)
    Ak = tp.ten_prod(uk, tp.ten_prod(f_diag(sk), vhk, dim_list, prod_type=prod_type, M=M), dim_list, prod_type=prod_type, M=M)
    err1 = norm(A - Ak) ** 2
    err2 = np.sum(s[k + 1:] ** 2)
    print('k = %d\terr diff = %0.2e' % (k + 1, abs(err1 - err2)))

    assert_array_almost_equal(statsk['err'], np.sqrt(err1))


# ==================================================================================================================== #
# check tsvdII
eps = np.finfo(np.float64).eps

shape_A = (15, 10, 3, 10)
M = create_orthogonal_m(shape_A[2:])
A = randn(*shape_A)
u, s, vh, stats = tp.ten_svdII(A, 1, dim_list, prod_type=prod_type, M=M, implicit_rank=None, compress_UV=True)

A_approx = tp.ten_prod(u, tp.ten_prod(f_diag(s), vh, dim_list, prod_type=prod_type, M=M), dim_list, prod_type=prod_type, M=M)
# assert_array_almost_equal(A, A_approx)
err = norm(A - A_approx) / norm(A)
print('full tsvdII relative error = %0.2e' % err)

nrm_Ahat2 = stats['nrm_Ahat'] ** 2
s_hat = tp.ten_transform(reshape(s, (1, *s.shape)), dim_list, prod_type=prod_type, M=M)[0]  # all singular values in transform domain

gamma = np.linspace(0.5, 1, 10)
for k in range(len(gamma)):
    uk, sk, vhk, statsk = tp.ten_svdII(A, gamma[k], dim_list, prod_type=prod_type, M=M, compress_UV=False)
    Ak = tp.ten_prod(uk, tp.ten_prod(f_diag(sk), vhk, dim_list, prod_type=prod_type, M=M), dim_list, prod_type=prod_type, M=M)

    # approximation error
    err1 = norm(A - Ak) ** 2

    # should be equal to the sum of the cutoff singular values in the transform domain
    sk_hat = tp.ten_transform(reshape(sk, (1, *sk.shape)), dim_list, prod_type=prod_type, M=M)[0]
    err2 = np.sum((s_hat[:sk_hat.shape[0]] - sk_hat) ** 2)  # only count singular values we did not store

    nrm_Ak2 = norm(Ak) ** 2
    assert_array_almost_equal(nrm_Ak2, np.sum(sk_hat ** 2))
    gamma_approx = nrm_Ak2 / nrm_Ahat2
    print('gamma = %0.2e\terr: %0.2e\tgamma diff: %0.2e' % (gamma[k], abs(err1 - err2), (gamma_approx - gamma[k])))

# implicit rank
for k in range(20):
    uk, sk, vhk, statsk = tp.ten_svdII(A, None, dim_list, prod_type=prod_type, M=M, compress_UV=False, implicit_rank=k + 1)
    Ak = tp.ten_prod(uk, tp.ten_prod(f_diag(sk), vhk, dim_list, prod_type=prod_type, M=M), dim_list, prod_type=prod_type, M=M)

    # approximation error
    err1 = norm(A - Ak) ** 2

    # should be equal to the sum of the cutoff singular values in the transform domain
    sk_hat = tp.ten_transform(reshape(sk, (1, *sk.shape)), dim_list, prod_type=prod_type, M=M)[0]
    err2 = np.sum((s_hat[:sk_hat.shape[0]] - sk_hat) ** 2)  # only count singular values we did not store

    nrm_Ak2 = norm(Ak) ** 2
    assert_array_almost_equal(nrm_Ak2, np.sum(sk_hat ** 2))
    gamma_approx = nrm_Ak2 / nrm_Ahat2
    print('r = %d\terr: %0.2e' % (k + 1, abs(err1 - err2)))