## GENERAL MATHEMATICAL OPS

import tensorflow as tf 
import numpy as np

tensor_zero_d = tf.constant(4)
print(tensor_zero_d)

tensor_one_d = tf.constant([2, 0, 10, 90.5])
print(tensor_one_d)

tensor_two_d = tf.constant([
    [1,2,3,4],
    [5,6,7,8]
])
print(tensor_two_d)

tensor_three_d = tf.constant([
    [
        [1,2,3,4],
        [5,6,7,8]
    ],
    [
        [1,2,3,4],
        [5,6,7,8]
    ]
])
print(tensor_three_d)

# GETTING DIMENSIONS:
tensor_three_d.ndim # single number

# GETTING SHAPE:
tensor_three_d.shape #(x, y, z, ...)

# DEFINING TENSOR AS FLOAT:
tensor_one_d_float = tf.constant([1,2,3,4,5], dtype=tf.float16)

# CASTING INT TO FLOAT:
tensor_one_d_int_float = tf.cast(tf.constant([0., 2, -3, 9], dtype=tf.float16), dtype=tf.int16)

# CASTING INT TO BOOL:
tensor_one_d_int_bool = tf.cast(tf.constant([0., 2, -3, 9], dtype=tf.float16), dtype=tf.bool) # 0 is False, everything else is True

# NUMPY:
np_array = np.array([1,2,4])
print(np_array)

converted_tensor = tf.convert_to_tensor(np_array)
print(converted_tensor)

# IDENTITY MATRIX
eye_tensor = tf.eye(
    num_rows=3,
    num_columns=None,
    batch_shape=None, # increasing this adds one more dimension to input so a 3*3 with batch_shape of 2 becomes a (2, 3, 3)
    dtype=tf.dtypes.float32,
    name=None
)

# FILL METHOD:
fill_tensor = tf.fill([2, 3], 9) # 2*3 matrix filled with 9s

# ONES LIKE TENSOR:
ones_like_tensor = tf.ones_like(fill_tensor) # returns an array of similar shape as input but filled with ones

# RANK:
tf.rank(fill_tensor) 

# SIZE
tf.size(fill_tensor, out_type=tf.float32) #total number of elements

# RANDOM TENSORS:
random_tensor = tf.random.normal(
    [3,2],
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

# UNIFORM NORMAL DISTRIBUTION:
random_uniform_tensor = tf.random.uniform(
    [5,],
    minval=0,
    maxval=100,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

# REPRODUCIBLE SEEDS: This allows you to create the same response consistently
tf.random.set_seed(5)
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10))
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10))
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10))
print(tf.random.uniform(shape=[3,], maxval=5, dtype=tf.int32, seed=10))

# TENSOR INDEXING
tensor_indexed = tf.constant([1,2,3,4,5])
print(tensor_indexed[0:4])
print(tensor_indexed[0:5:2]) #[minIndex:maxIndex:step]
print(tensor_indexed[2:-1]) #second item all the way to last but 1 item

tf.range(2, 6) # generate natural numbers from 2 to 5

print(tensor_two_d[0:3, 0:2]) #first three rows, first two columns. 
print(tensor_two_d[0:3, :]) #first three rows, all columns

# ABSOLUTE VALUE OPS
tf.abs(tf.constant(-0.2))
tf.abs(tf.constant([-2.24 + 4.5j])) # results in a single number represented by sqrt of a^2 + b^2

# ELEMENTWISE OPS
tf.add([1,2], [3,4]) # for irregularly shaped tensors, the smaller tensor is stretched out to match the shape of the bigger one.
tf.subtract([1,2], [3,4])
tf.multiply([1,2], [3,4]) # a (1,6) tensor multiplying a (3,1) tensor forces the (1, 6) to become (3,6)
tf.math.divide_no_nan([1,2], [3,4])  

# ELEMENTWISE MAX&MIN OF TWO TENSORS
tf.math.maximum()
tf.math.minimum()
tf.math.argmax(tensor_two_d) # returns the index of the maximum value
tf.math.argmax(tensor_two_d, 0) # this is for 2+ dimensional tensors. it fixes the axis on the row and returns index of max val in each column. a (3,5) matrix returns a (1,5) response

tf.pow(tf.constant([2,4]), tf.constant([3,2])) # [8, 16]

tf.math.reduce_sum( #reduce_max returns the max value of the input 
    [[1,2,3], [4,5,6]],
    axis=None, # None means sum everything. axis 0 means fix on the rows and sum each column
    keepdims=False,
    name=None
)

tf.math.sigmoid() # normalizes data over a large range into -1<x<1
tf.math.top_k(tensor_two_d, k = 2) # fetches the top 2 per row since k is 2