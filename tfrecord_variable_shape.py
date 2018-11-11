import tensorflow as tf
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = '/tmp/tfrecord_variable_shape.tfrecords'
graph_dir = '/tmp/tfrecord_variable_shape_graph'
n_records = 10
max_n_dim = 4
rank = 2
n_classes = 3
input_dicts = []
for ii in range(n_records):
    shape = np.random.randint(1, max_n_dim + 1, rank)
    inputs = np.random.rand(*shape)
    inputs_as_bytes = inputs.astype(np.float32).tostring()
    input_dicts.append({'inputs': inputs_as_bytes, 'shape': shape})
targets = np.random.randint(0, n_classes, n_records)

""" Write tfrecords file """
writer = tf.python_io.TFRecordWriter(tfrecord_filename)
for i, t in zip(input_dicts, targets):
    feature = {
        'input_shape': _int64_feature(i['shape']),
        'inputs': _bytes_feature(tf.compat.as_bytes(i['inputs'])),
        'targets': _int64_feature([t]),
        }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()


""" Read tfrecords file """
def _parse_fn(example_proto):
    features = {
            'input_shape': tf.FixedLenFeature([rank], tf.int64),
            'inputs': tf.FixedLenFeature([], tf.string),
            'targets': tf.FixedLenFeature([], tf.int64),
            }
    parsed_features = tf.parse_single_example(example_proto, features=features)
    flat_inputs = tf.decode_raw(parsed_features['inputs'], tf.float32)
    shaped_inputs = tf.reshape(flat_inputs, parsed_features['input_shape'])
    targets = tf.cast(parsed_features['targets'], tf.int32, name='targets')
    return shaped_inputs, targets

dataset = tf.data.TFRecordDataset([tfrecord_filename])
dataset = dataset.map(_parse_fn)
iterator = dataset.make_one_shot_iterator()

inputs, targets = iterator.get_next()
with tf.Session() as sess:
    # writer = tf.summary.FileWriter(graph_dir, sess.graph)
    print(sess.run(inputs))
    print(sess.run(targets))
    # writer.close()
