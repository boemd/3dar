import matplotlib.image as img

import numpy as np
import logging
import keras
import tensorflow as tf
import os
import scipy.io as sio
from os import scandir

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', 'C:/Users/gabro/Documents/3D/Dataset/db_train_set', 'Input directory')
tf.flags.DEFINE_string('test_dir', 'C:/Users/gabro/Documents/3D/Dataset/db_test_set', 'Test directory')
tf.flags.DEFINE_string('val_dir', 'C:/Users/gabro/Documents/3D/Dataset/db_val_set', 'Test directory')


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

def image_matrix_creation_small(images_paths1):
    image_matrix = np.zeros((len(images_paths1), 128, 128))
    for i in range(len(images_paths1)):
            image_matrix[i, :, :] = img.imread(images_paths1[i])
    return image_matrix

def image_matrix_creation(images_paths1, images_paths2):
    image_matrix = np.zeros((len(images_paths1), 128, 128, 2))
    for i in range(len(images_paths1)):
            image_matrix[i, :, :, 0] = img.imread(images_paths1[i])
            image_matrix[i, :, :, 1] = img.imread(images_paths2[i])
    return image_matrix


def mat_matrix_creation(mat_path):
    mat_matrix = np.zeros((len(mat_path), 8))
    for i in range(len(mat_path)):
        mat_matrix[i] = sio.loadmat(mat_path[i])['out'].T
    return mat_matrix


if __name__ == '__main__':
    a_val, b_val, mat_val = data_reader(FLAGS.test_dir)
    x_val = image_matrix_creation(a_val, b_val)
    np.save('x_test', x_val)
    y_val = mat_matrix_creation(mat_val)
    np.save('y_test', y_val)
