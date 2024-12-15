import tensorflow as tf 
import numpy as np

a = tf.constant([[2,3],[4,5],[6,7]])
b = tf.constant([[2,3],[4,5],[6,7]])

#expand dimensions:
print(tf.expand_dims(a, axis=0)) # adds one dimension to overall so a 3,1,2 array becomes 1,3,1,2
print(tf.expand_dims(a, axis=1)) # adds one dimension after the first so a 3,1,2 array becomes 3,1,1,2 (can expand to axis=2, etc.)

# reduce dimensions:
print(tf.squeeze(a, axis=1)) # this will not work however since you can only squeeze a dimension of length 1

# reshaping a tensor:
print(tf.reshape(a, tf.constant([2,3]).shape)) # total number of values must be same as that of the original array
print(tf.reshape(a, tf.constant([2,-1]).shape)) # the negative allows tensorflow to resize the final dimension based on the shape if you only know one dimension

# concat:
print(tf.concat([a, a], axis=0)) #adds dims of rows and maintains same column value as dimension of result

# stack:
print(tf.stack([a, b], axis=0)) #given two vectors, this means a&b with dims [3,2] results in [2,3,2]

# pad

# gather
print(tf.gather(a, params=tf.range(1,4))) #get index 1,2,3 if a is one dimensional
print(tf.gather(a, params=[0, 3], axis=0)) #get row 0 and 3 given that axis is 0(row) if a is 2 dimensional
# given a 1,4,3 item with axis of 1 and params of 2,0, it means pick the third and first rows on the first axis(one with dim of 4)
# if it is 2,4,3, it does the above but twice the operation since the first dimension is now 2

# gather_nd: strictly fetches at the indexes specified for a
print(tf.gather_nd(a, [[2, 1]])) # gets the second row and first column in the vector <a> for the item at default ie. axis 0.
#[2,1] instead of [[2,1]] does same since we only have one item at axis 0

indices = [[0, 1], [1, 0]]
params = [[[1,2], [3,4]],[[5,6], [7,8]]]
tf.gather_nd(params, indices) # returns [[3,4], [5,6]]
tf.gather_nd(params, indices, batch_dims=1) # batch aware so it treats each group as a batch hence [2,7]. Also, a 2,3,4 will need at least 3 descriptive indices. 

#ragged tensors
tf.ragged.constant(a) #has a shape of [rows, None] since when used on vectors that do not have the same length

# sparse and dense tensors
# tf.device(param: 'CPU'|'GPU') 