import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


def _parse_example(serialized_example):
    """
    Parse a serialized example
    """
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
    # concatenates the images with shape (N, N, 1) and produces a tensor of shape (N, N, 2)
    concat = tf.concat([image_A, image_B], axis=2)
    return concat, homography_list


def get_dataset(tfrecords_path, batch_size=32):
    """
    Parse a tf.Data.TFRecordDataset from the serialized examples
    :param tfrecords_path: path of the TFRecord file
    :param batch_size: size of the batch
    :return: batch of data and size of the dataset
    """
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(_parse_example, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.shuffle(buffer_size=200000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    #_plot_batch(x)

    size = 0
    for record in tf.python_io.tf_record_iterator(tfrecords_path):
        size += 1

    return x/255, y, size


def get_dataset_iterator(tfrecords_path, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(_parse_example, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.shuffle(buffer_size=200000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def _plot_batch(batch, batch_size):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    b = sess.run(batch)
    plt.figure(1)
    for i in range(batch_size):
        xa = b[i, :, :, 0]
        xb = b[i, :, :, 1]
        xa = np.squeeze(xa)
        xb = np.squeeze(xb)
        plt.subplot(121)
        plt.imshow(xa)
        plt.subplot(122)
        plt.imshow(xb)

        plt.show()
    sess.close()
