import tensorflow as tf

rank0_tensor = tf.Variable("just one string", tf.string)
rank1_tensor = tf.Variable(['small', 'list'], tf.string)
rank2_tensor = tf.Variable([['one', 'list'], ['inside', 'the other']], tf.string)

# numpy is the answer - dimensions
tf.rank(rank0_tensor)
tf.rank(rank1_tensor)
tf.rank(rank2_tensor)

# how many items in each dimension
print(rank2_tensor.shape)

tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
tensor3 = tf.reshape(tensor2, [3, -1])

print(tensor1, tensor2, tensor3)


rank2_tensor.numpy()
