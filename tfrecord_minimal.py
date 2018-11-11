import tensorflow as tf
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

tfrecord_filename = '/tmp/tfrecord_minimal.tfrecords'
graph_dir = '/tmp/tfrecord_minimal_graph'
n_records = 10
feature_dim = 4
n_classes = 3
inputs = np.random.rand(n_records, feature_dim)
targets = np.random.randint(0, n_classes, n_records)

""" Write tfrecords file """
writer = tf.python_io.TFRecordWriter(tfrecord_filename)
for i, t in zip(inputs, targets):
    feature = {
        'inputs': _float_feature(i),
        'targets': _int64_feature([t]),
        }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()


""" Read tfrecords file """
def _parse_fn(example_proto):
    features = {
            'inputs': tf.FixedLenFeature([feature_dim], tf.float32),
            'targets': tf.FixedLenFeature([], tf.int64),
            }
    parsed_features = tf.parse_single_example(example_proto, features=features)
    inputs = tf.cast(parsed_features['inputs'], tf.float32, name='inputs')
    targets = tf.cast(parsed_features['targets'], tf.int32, name='targets')
    return inputs, targets

dataset = tf.data.TFRecordDataset([tfrecord_filename])
"""
at this point, the dataset contains serialized examples. So if were to print
a single element, it would look something like this:
b'\n.\n\x10\n\x07targets\x12\x05\x1a\x03\n\x01\x01\n\x1a\n\x06inputs\x12\x10\x12\x0e\n\x0c\xc5[\xd9=x(\xc9>\x98\xf8k?'
"""
dataset = dataset.map(_parse_fn)
iterator = dataset.make_one_shot_iterator()
inputs, targets = iterator.get_next()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(graph_dir, sess.graph)
    print(sess.run(inputs))
    print(sess.run(targets))
    writer.close()
