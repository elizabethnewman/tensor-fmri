import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.random import permutation
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.utils as tu
from tensor.tucker import hosvd, tucker_product

# np.random.seed(20)

shape_A = (20, 20, 20, 20, 20)
A = randn(*shape_A)
# A = A / norm(A)

# higher tolerance means worse approximation, but more compression
tol = 1e-2

dim_order = permutation(np.arange(len(shape_A)))

G, U, ranks = hosvd(A, tol, dim_order=dim_order, sequential=False, ranks=None)

Ak = tucker_product(G, U)

err = norm(A - Ak) / norm(A)

print('dim order: ', dim_order)
print('shape: ', shape_A)
print('ranks: ', ranks)
print('hosvd: error = %0.6e' % err)
print('tol: tol = %0.2e' % tol)
print('check tolerance: %d' % (err < tol))


# force


