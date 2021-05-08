import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.utils as tu


# ==================================================================================================================== #
# test reshape orientation
shape_A = (2, 3, 4, 5)
A_true = np.zeros(shape_A)

count = 0
for i3 in range(shape_A[3]):
    for i2 in range(shape_A[2]):
        for i1 in range(shape_A[1]):
            for i0 in range(shape_A[0]):
                # count first dimension first
                A_true[i0, i1, i2, i3] = count
                count += 1

a = np.arange(count)
A_hat = tu.reshape(a, shape_A)

assert_array_equal(A_hat, A_true)

# ==================================================================================================================== #
# test reshape undo
shape_A = (2, 3, 4, 5)
shape_A_new = (6, 20)

A_true = randn(*shape_A)
A_new = tu.reshape(A_true, shape_A_new)
A_hat = tu.reshape(A_new, shape_A)

assert_array_equal(A_hat, A_true)

# ==================================================================================================================== #
# test fold and unfold

shape_A = (2, 3, 4, 5, 6)

A = randn(*shape_A)

for i in range(len(shape_A)):
    A_unfold = tu.unfold(A, dim=i)
    A_fold = tu.fold(A_unfold, shape_A, dim=i)
    assert_array_equal(A, A_fold)

A_unfold = tu.unfold(A, dim=1)

A_mat = np.zeros(A_unfold.shape)
for i in range(A_mat.shape[1]):
    A_mat[:, i] = tu.reshape(A[:, i], -1)

assert_array_equal(A_unfold, A_mat)








