import tensorflow as tf
import numpy as np


dataset = tf.data.Dataset.from_tensor_slices({
    'input_float': [[1.2], [2.4], [3.6]],
    'input_string': [['string1'], ['string2'], ['string3']],
    'targets': [1, 0, 1]})
iterator = dataset.batch(2).make_one_shot_iterator()
features = iterator.get_next()
numerical_column = tf.feature_column.numeric_column(
    key='input_float',
    shape=(1),
    default_value=0,
    dtype=tf.float32)
string_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='input_string',
        vocabulary_list=['string1', 'string2', 'string3'])
string_indicator = tf.feature_column.indicator_column(string_categorical)
input_layer = tf.feature_column.input_layer(
    features,
    feature_columns=[numerical_column, string_indicator])
targets = features['targets']

with tf.Session() as sess:
    sess.run(tf.tables_initializer()) # required for categorical columnns
    print(sess.run((input_layer, targets)))
    print(sess.run((input_layer, targets)))
