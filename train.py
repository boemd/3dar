import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from datetime import datetime
import model

FLAGS = tf.flags.FLAGS
CORES = multiprocessing.cpu_count()

tf.flags.DEFINE_string('tfrecords_train_path', 'data/tfrecords/db_train.tfrecords',
                       'path to the training set (.tfrecords)')
tf.flags.DEFINE_string('tfrecords_validation_path', 'data/tfrecords/db_val.tfrecords',
                       'path to the validation set (.tfrecords)')
tf.flags.DEFINE_string('tfrecords_test_path', 'data/tfrecords/db_test.tfrecords',
                       'path to the test set (.tfrecords)')
tf.flags.DEFINE_integer('batch_size', 3, 'size of the batch, default: 54')
tf.flags.DEFINE_integer('shuffle_buffer_size', 500100, 'size of the shuffle buffer, default 500100')
tf.flags.DEFINE_float('initial_learning_rate', 5e-3, 'initial learning rate, default: 0.005')
tf.flags.DEFINE_float('sgd_momentum', 9e-1, 'learning rate momentum, default: 0.9')
tf.flags.DEFINE_integer('training_epochs', 1, 'number of training iterations over the entire dataset, default: 12')
tf.flags.DEFINE_integer('val_steps', 2,
                        'specifies how many training epochs to run before a new validation run is performed, default: 2')


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


def get_dataset(tfrecords_path):
    """
    Parse a tf.Data.TFRecordDataset from the serialized examples
    :param tfrecords_path: path of the TFRecord file
    :return: batch of data and size of the dataset
    """
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(_parse_example, num_parallel_calls=CORES)
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    #_plot_batch(x)

    size = 0
    for record in tf.python_io.tf_record_iterator(tfrecords_path):
        size += 1

    return x, y, size


def get_dataset_iterator(tfrecords_path):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(_parse_example, num_parallel_calls=CORES)
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def _plot_batch(batch):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    b = sess.run(batch)
    plt.figure(1)
    for i in range(FLAGS.batch_size):
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


def train():
    model_name = 'model_' + datetime.now().strftime('%Y%m%d-%H%M') + '.h5'

    x_train, y_train, sz_train = get_dataset(FLAGS.tfrecords_train_path)
    x_val, y_val, sz_val = get_dataset(FLAGS.tfrecords_validation_path)
    x_test, y_test, sz_test = get_dataset(FLAGS.tfrecords_test_path)

    batch_size = FLAGS.batch_size
    net = model.HomographyNet(name='h_net', batch_size=batch_size)
    net.build_model()

    loss, mtr = net.train_model(base_lr=FLAGS.initial_learning_rate,
                                loss = 'categorical_crossentropy',
                                momentum=FLAGS.sgd_momentum,
                                epochs=FLAGS.training_epochs,
                                train_data=(x_train, y_train),
                                val_data=(x_val, y_val),
                                test_data=(x_test, y_test),
                                steps_per_epoch=int(sz_train / batch_size),
                                val_steps=FLAGS.val_steps)
    net.save_model('models/'+model_name)
    print('Test loss:', loss)
    print('Test msle', mtr)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()


