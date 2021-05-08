import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.mode_k as mode_k
from tensor.utils import reshape


# ==================================================================================================================== #
# modek one dimension
shape_A = (2, 3, 4, 5, 6)
k = 4
A = randn(*shape_A)
M = randn(10, A.shape[k])

A_hat_true = np.moveaxis(A, k, 0)
shape_A_tmp = A_hat_true.shape

A_hat_true = reshape(A_hat_true, (A.shape[k], -1))
A_hat_true = M @ A_hat_true
A_hat_true = reshape(A_hat_true, (M.shape[0], *shape_A_tmp[1:]))
A_hat_true = np.moveaxis(A_hat_true, 0, k)

A_hat = mode_k.modek_product_one_dimension(A, M, k)

assert_array_almost_equal(A_hat, A_hat_true)
err = norm(A_hat - A_hat_true) / norm(A_hat_true)
print('modek: relative error = %0.2e' % err)

k = 3
A = randn(*shape_A)
M = np.ones((1, A.shape[k]))

A_hat_true = np.sum(A, axis=k, keepdims=True)

A_hat = mode_k.modek_product_one_dimension(A, M, k)

assert_array_almost_equal(A_hat, A_hat_true)
err = norm(A_hat - A_hat_true) / norm(A_hat_true)
print('modek sum: relative error = %0.2e' % err)


# ==================================================================================================================== #
# modek

shape_A = (2, 3, 4, 5, 6)
k = 3
A = randn(*shape_A)
M = randn(7, A.shape[k])

A1 = mode_k.modek_product_one_dimension(A, M, k)
A2 = mode_k.modek_product(A, M, k)

assert_array_almost_equal(A1, A2)
err = norm(A1 - A2) / norm(A1)
print('modek: relative error = %0.2e' % err)

# ==================================================================================================================== #
# modek many
shape_A = (2, 3, 4, 5, 6)
k = 3
A = randn(*shape_A)
M = [None, None, None, randn(7, A.shape[3]), randn(10, A.shape[4])]
A1 = A
for i in range(len(M)):
    if M[i] is not None:
        A1 = mode_k.modek_product_one_dimension(A1, M[i], i)

A2 = A
idx = np.random.permutation(len(M))
for i in idx:
    if M[i] is not None:
        A2 = mode_k.modek_product_one_dimension(A2, M[i], i)


assert_array_almost_equal(A1, A2)
err = norm(A1 - A2) / norm(A1)
print('modek reorder: relative error = %0.2e' % err)

A3 = mode_k.modek_product(A, M)

assert_array_almost_equal(A1, A3)
err = norm(A1 - A3) / norm(A1)
print('modek many: relative error = %0.2e' % err)

idx = np.random.permutation(len(M))
M = [M[i] for i in idx]
A3 = mode_k.modek_product(A, M, axis=idx)

assert_array_almost_equal(A1, A3)
err = norm(A1 - A3) / norm(A1)
print('modek many reorder: relative error = %0.2e' % err)


M = [None, None, randn(7, A.shape[2]), None, randn(10, A.shape[4])]
A1 = mode_k.modek_product(A, M)

A4 = mode_k.modek_product(A, (M[2], M[4]), axis=(2, 4))
assert_array_almost_equal(A1, A4)
err = norm(A1 - A4) / norm(A1)
print('modek partial axis: relative error = %0.2e' % err)

# ==================================================================================================================== #
# fold and unfold
shape_A = (2, 3, 1, 4, 5)
A = randn(*shape_A)

k = 3

A_mat = mode_k.modek_unfold(A, k)
A_ten = mode_k.modek_fold(A_mat, k, A.shape)

assert_array_equal(A_ten, A)
err = norm(A_ten - A) / norm(A)
print('modek fold/unfold: relative error = %0.2e' % err)

# random selection and demonstration of indexing of mode_k unfold and fold
assert_array_equal(A_mat[:, 0], A[0, 0, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 1], A[1, 0, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 2], A[0, 1, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 3], A[1, 1, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 4], A[0, 2, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 5], A[1, 2, 0, :, 0].reshape(-1))
assert_array_equal(A_mat[:, 6], A[0, 0, 0, :, 1].reshape(-1))
assert_array_equal(A_mat[:, 7], A[1, 0, 0, :, 1].reshape(-1))
assert_array_equal(A_mat[:, 8], A[0, 1, 0, :, 1].reshape(-1))
assert_array_equal(A_mat[:, 9], A[1, 1, 0, :, 1].reshape(-1))
