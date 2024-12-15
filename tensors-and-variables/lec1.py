# MATRIX ALGEBRA:

import tensorflow as tf 
import numpy as np

a = tf.constant([[2,3],[4,5],[6,7]])
b = tf.constant([[8,9],[10,11],[12,13]])
# unlike tf.math.multiply which is an element-wise operation, tf.math.linalg is actual matrix multiplication
tf.linalg.matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False, # a mathematical op you can perform but idk full meaning yet.
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    grad_a=False,
    grad_b=False,
    name=None
)

print(a@b) #matrix multiplication
print(a*b) #element-wise multiplication

# TRANSPOSE:

print(tf.transpose(a)) # or tf.linalg.matrix_transpose

# BAND=PART MATRIXES:

tf.linalg.band_part(input, 0, -1) # Upper triangular matrix
tf.linalg.band_part(input, -1, 0) # Lower triangular matrix
tf.linalg.band_part(input, 0, 0) # Diagonal matrix

# INVERSE:

tensor_two_d = tf.constant([[1,-2, 0], [3, 5, 100], [1,5,6]])

tf.linalg.inv(tensor_two_d )

s, u, v = tf.linalg.svd(tensor_two_d)

# einsum operator in place of matmul as an example:
A = np.array([[2,3,4,5],
              [2,3,4,5],
              [2,3,4,5]])

B = np.array([[2,3,4,5,6],
              [2,3,4,5,6],
              [2,3,4,5,6],
              [2,3,4,5,6]])

print(np.einsum("ij, jk -> ik", A, B)) #same as np.matmul(A,B)
print(np.einsum("ij, ij -> ij", A, B)) #same as A*B
print(np.einsum("ij -> ji"), A) #Transpose A ie. A.Y
print(np.einsum("bij, bjk -> bik", A, B)) #3D array multiplication
print(np.einsum("bij -> ", A)) #sums all elements of A. output is empty, indicating shape of 0 ie. constant
print(np.einsum("ij -> i", A)) #sums all elements of A and returns vector of size i

A = np.random.randn(2,4,4,2) #bcij
B = np.random.randn(2,4,4,1) #bcik
print(np.einsum("bcik, bcij -> bckj", A, B)) # does the transform of B and then multiplies by A
np.matmul(np.transpose(B, (0,1,3,2)), A) # same as einsum above