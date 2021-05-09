import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from numpy.random import permutation
from numpy.testing import assert_array_almost_equal, assert_array_equal
import tensor.utils as tu
from tensor.tensor_train import ttsvd, tt_product

# np.random.seed(20)

shape_A = (3, 4, 5, 6, 7)
A = randn(*shape_A)
A = A / norm(A)

# higher tolerance means worse approximation, but more compression
tol = 0
dim_order = permutation(np.arange(len(shape_A)))

G, ranks = ttsvd(A, tol, dim_order=dim_order, ranks=None)

Ak = tt_product(G, shape_A, dim_order=dim_order)


err = norm(A - Ak) / norm(A)

print('dim order: ', dim_order)
print('shape: ', shape_A)
print('ranks: ', ranks)
print('ttsvd: error = %0.6e' % err)
print('tol: tol = %0.2e' % tol)
print('check tolerance: %d' % (err < tol))
