import numpy as np
import tensorflow as tf

n_examples = 100
n_features = 3
np_features = {
    'input1': np.random.rand(n_examples, n_features),
    'input2': np.random.rand(n_examples, n_features)}
dataset = tf.data.Dataset.from_tensor_slices(np_features)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))
