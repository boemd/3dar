import tensorflow as tf


def conv_layer(input, num_filters, filter_size, reuse=False, is_training=True, name='conv0'):

    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("weights",
                           shape=[filter_size, filter_size, input.get_shape()[3], num_filters])

        conv = tf.nn.conv2d(input, weights, padding='SAME', strides=[1, 2, 2, 1])

        normalized = _batch_norm(conv, is_training)

        output = tf.nn.relu(normalized)

        return output


def pool_layer(input, pool_size, strides, name='pool0'):
    return tf.layers.max_pooling2d(input, pool_size, strides, padding='valid', data_format='channels_last', name=name)


def flatten_layer(input):
    return tf.contrib.layers.flatten(input)


def dropout_layer(input, keep_prob=0.5, is_training=True):
    return tf.contrib.layers.dropout(input, keep_prob, is_training)


def dense_layer(input, out_dim, activation=tf.nn.relu):
    return tf.layers.dense(input, units=out_dim, activation=activation)


def _weights(name, shape, mean=0.0, stddev=0.02):
    """ Helper to create an initialized Variable
    Args:
      name: name of the variable
      shape: list of ints
      mean: mean of a Gaussian
      stddev: standard deviation of a Gaussian
    Returns:
      A trainable variable
    """
    var = tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(
            mean=mean, stddev=stddev, dtype=tf.float32))
    return var


def _batch_norm(input, is_training):
    """ Batch Normalization
    """
    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_norm(input,
                                            decay=0.9,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=is_training)

