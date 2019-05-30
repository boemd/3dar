import tensorflow as tf
import ops


class HomographyNet00:

    def __init__(self, name, is_training):
        self.name = name
        self.reuse = False
        self.is_training = is_training

    def __call__(self, input):
        """
        Args:
          input: batch_size x width x height x 3
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name):
            c1 = ops.conv_layer(input, 64, 3, reuse=self.reuse, is_training=self.is_training, name='conv1')
            c2 = ops.conv_layer(c1, 64, 3, reuse=self.reuse, is_training=self.is_training, name='conv2')
            p1 = ops.pool_layer(c2, 2, strides=[2, 2], name='pool1')
            c3 = ops.conv_layer(p1, 64, 3, reuse=self.reuse, is_training=self.is_training, name='conv3')
            c4 = ops.conv_layer(c3, 64, 3, reuse=self.reuse, is_training=self.is_training, name='conv4')
            p2 = ops.pool_layer(c4, 2, strides=[2, 2], name='pool2')
            c5 = ops.conv_layer(p2, 128, 3, reuse=self.reuse, is_training=self.is_training, name='conv5')
            c6 = ops.conv_layer(c5, 128, 3, reuse=self.reuse, is_training=self.is_training, name='conv6')
            p3 = ops.pool_layer(c6, 2, strides=[2, 2], name='pool3')
            c7 = ops.conv_layer(p3, 128, 3, reuse=self.reuse, is_training=self.is_training, name='conv7')
            c8 = ops.conv_layer(c7, 128, 3, reuse=self.reuse, is_training=self.is_training, name='conv8')
            f = ops.flatten_layer(c8)
            dr1 = ops.dropout_layer(f, 0.5, is_training=self.is_training)
            d1 = ops.dense_layer(dr1, 1024)
            dr2 = ops.dropout_layer(d1, 0.5, is_training=self.is_training)
            d2 = ops.dense_layer(dr2, 1024)

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return d2

    def sample(self, input):
        h = self.__call__(input)
        return h

