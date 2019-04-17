import tensorflow as tf
import logging
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('tfrecords_train_path', 'data/tfrecords/test.tfrecords',
                       'path to the training set (.tfrecords)')
tf.flags.DEFINE_integer('batch_size', 16, 'size of the batch, default: 16')
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
            'homography/homography_list': tf.FixedLenFeature([], tf.float32)
        })
    encoded_image_A = features['image/encoded_A_image']
    image_A = tf.image.decode_png(encoded_image_A)
    encoded_image_B = features['image/encoded_B_image']
    image_B = tf.image.decode_png(encoded_image_B)
    homography_list = features['homography/homography_list']

    return image_A, image_B, homography_list


def train():
    dataset = tf.data.TFRecordDataset(FLAGS.tfrecords_train_path)
    dataset = dataset.map(_parse_example, num_parallel_calls=4)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    for step in range(FLAGS. training_steps):
        image_A, image_B, homography = iterator.get_next()
        plt.imshow(image_B)


    return


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
