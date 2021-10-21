from numpy.random import randn
from numpy.testing import assert_array_almost_equal
from scipy.linalg import dft
import tensor.tensor_product_wrapper as tp


# create tensors
shape_A = (2, 3, 4, 5)
shape_B = (shape_A[1], 2, *shape_A[2:])
A = randn(*shape_A)
B = randn(*shape_B)

# multiply with the t-product
C1 = tp.ten_prod(A, B, prod_type='t')

# multiply with the m-product
M = []
for i in range(len(shape_A) - 2):
    Mi = dft(shape_A[i + 2], 'sqrtn')
    M.append(Mi)

C2 = tp.ten_prod(A, B, M=M, prod_type='m')

assert_array_almost_equal(C1, C2)
