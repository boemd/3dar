import tensorflow as tf
import os
import scipy.io
import numpy as np
from os import scandir

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', '../Dataset/db_train_set', 'Input directory')
tf.flags.DEFINE_string('output_file', 'data/tfrecords/db_train.tfrecords', 'Output tfrecords file')


def data_reader(input_dir):
    """
    scans the input folder and organizes the various paths
    :param input_dir: directory containing images (.png) of type A and B and their respective homography matrices (.mat)
    :return: lists of paths of images (types A and B) and homography matrices
    """
    images_A_paths = []
    images_B_paths = []
    homographies_paths = []

    for file in scandir(input_dir):
        if file.name.endswith('A.png'):
            images_A_paths.append(file.path)
        elif file.name.endswith('B.png'):
            images_B_paths.append(file.path)
        elif file.name.endswith('.mat'):
            homographies_paths.append(file.path)

    cond_1 = len(images_A_paths) != len(images_B_paths)
    cond_2 = len(images_B_paths) != len(homographies_paths)
    cond_3 = len(homographies_paths) != len(images_A_paths)

    # check correct correspondences between lists
    if cond_1 or cond_2 or cond_3:
        raise Exception('Paths not corresponding. Length mismatch.')

    for i in range(len(homographies_paths)):
        a = images_A_paths[i].split('\\')[-1].split('A')[0]
        b = images_B_paths[i].split('\\')[-1].split('B')[0]
        c = homographies_paths[i].split('\\')[-1].split('.')[0]
        if a != b or b != c or c != a:
            raise Exception('Paths not corresponding. Not corresponding files.')

    # everything is matched
    return images_A_paths, images_B_paths, homographies_paths


def data_writer(input_dir, output_file):
    """
    serializes the data
    :param input_dir: directory containing images (.png) of type A and B and their respective homography matrices (.mat)
    :param output_file: output tfrecords file path
    :return: data of the input_dir serialized in a tfrecords file
    """
    images_A_paths, images_B_paths, homographies_paths = data_reader(input_dir)

    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error as e:
        pass

    num_images = len(images_A_paths)

    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(num_images):
        image_A_path = images_A_paths[i]
        image_B_path = images_B_paths[i]
        homography_path = homographies_paths[i]

        with tf.gfile.FastGFile(image_A_path, 'rb') as f:
            image_A = f.read()

        with tf.gfile.FastGFile(image_B_path, 'rb') as f:
            image_B = f.read()

        homography_list = scipy.io.loadmat(homography_path).get('out').tolist()
        homography = [val for sublist in homography_list for val in sublist]

        example = convert_to_example(image_A_path, image_A, image_B_path, image_B, homography_path, homography)
        writer.write(example.SerializeToString())

        if i % 1000 == 0:
            print("Processed {}/{}.".format(i, num_images))

    print("Done.")
    writer.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_to_example(image_A_path, image_A, image_B_path, image_B, homography_path, homography):
    """
    an example containing data and paths is serialized
    """
    image_A_name = image_A_path.split('/')[-1]
    image_B_name = image_B_path.split('/')[-1]
    homography_name = homography_path.split('/')[-1]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/image_A_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(image_A_name))),
        'image/encoded_A_image': _bytes_feature(image_A),
        'image/image_B_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(image_B_name))),
        'image/encoded_B_image': _bytes_feature(image_B),
        'homography/homography_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(homography_name))),
        'homography/homography_list': _float_feature(homography)
    }))

    return example


def main(unused_argv):
    data_writer(FLAGS.input_dir, FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
