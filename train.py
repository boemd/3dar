import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from tensorflow.python import keras
import model

FLAGS = tf.flags.FLAGS
CORES = multiprocessing.cpu_count()
tf.flags.DEFINE_string('tfrecords_train_path', 'data/tfrecords/test.tfrecords',
                       'path to the training set (.tfrecords)')
tf.flags.DEFINE_integer('batch_size', 16, 'size of the batch, default: 16')
tf.flags.DEFINE_integer('shuffle_buffer_size', 500100, 'size of the shuffle buffer, default 500100')
tf.flags.DEFINE_integer('training_steps', 1, 'number of training iterations, default: 100')


def _parse_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/image_A_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_A_image': tf.FixedLenFeature([], tf.string),
            'image/image_B_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_B_image': tf.FixedLenFeature([], tf.string),
            'homography/homography_name': tf.FixedLenFeature([], tf.string),
            'homography/homography_list': tf.FixedLenFeature((8,), tf.float32)
        })
    encoded_image_A = features['image/encoded_A_image']
    image_A = tf.image.decode_png(encoded_image_A)
    encoded_image_B = features['image/encoded_B_image']
    image_B = tf.image.decode_png(encoded_image_B)
    homography_list = features['homography/homography_list']
    stack = tf.stack([image_A, image_B], axis=3)

    return stack, homography_list


def get_dataset():
    dataset = tf.data.TFRecordDataset(FLAGS.tfrecords_train_path)
    dataset = dataset.map(_parse_example, num_parallel_calls=CORES)
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    #_plot_batch(x)
    return x, y


def _plot_batch(batch):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    b = sess.run(batch)
    plt.figure(1)
    for i in range(FLAGS.batch_size):
        xa = b[i, :, :, :, 0]
        xb = b[i, :, :, :, 1]
        xa = np.squeeze(xa)
        xb = np.squeeze(xb)
        plt.subplot(121)
        plt.imshow(xa)
        plt.subplot(122)
        plt.imshow(xb)

        plt.show()
    sess.close()


def train():
    x_batch, y_batch = get_dataset()
    x_input = keras.layers.Input(tensor=x_batch)
    net = model.HomographyNet.build_model()
    return

def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()


