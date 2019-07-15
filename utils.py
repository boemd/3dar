import sklearn.metrics as sk_metrics
import numpy as np
from sequential_generator import DataGenerator
from os import scandir


def mse(true, predicted):
    return sk_metrics.mean_squared_error(true.T, predicted.T, multioutput='raw_values')


def mean_mse(true, predicted):
    error = sk_metrics.mean_squared_error(true.T, predicted.T, multioutput='raw_values')
    mean = np.mean(error)
    return mean


def data_reader(input_dir, extA='A.png', extB='B.png', extL='M.mat'):
    """
    Scans the input folder and organizes the various paths
    :param input_dir: directory containing images of type A and B and their respective output files
    :return: lists of paths of images (types A and B) and output array
    """
    images_A_paths = []
    images_B_paths = []
    homographies_paths = []

    for file in scandir(input_dir):
        if file.name.endswith(extA):
            images_A_paths.append(file.path)
        elif file.name.endswith(extB):
            images_B_paths.append(file.path)
        elif file.name.endswith(extL):
            homographies_paths.append(file.path)

    cond_1 = len(images_A_paths) != len(images_B_paths)
    cond_2 = len(images_B_paths) != len(homographies_paths)
    cond_3 = len(homographies_paths) != len(images_A_paths)

    # Check correct correspondences between lists
    if cond_1 or cond_2 or cond_3:
        raise Exception('Paths not corresponding. Length mismatch.')

    for i in range(len(homographies_paths)):
        a = images_A_paths[i].split('\\')[-1].split('A')[0]
        b = images_B_paths[i].split('\\')[-1].split('B')[0]
        c = homographies_paths[i].split('\\')[-1].split('.')[0]

    # Everything is matched
    return images_A_paths, images_B_paths, homographies_paths


def lr_callback(epochs, lr):
    updated_lr = lr
    if ((epochs+1) % 6) == 0:
        updated_lr /= 10
    return updated_lr


def create_seq_generator(directory, batch_size, extA='A.png', extB='B.png', extL='M.mat'):
    """
    Create generators
    :param directory: directory of the files
    :param batch_size: dimension of the batch
    :return:
    """
    a, b, mat = data_reader(directory, extA, extB, extL)
    my_batch_generator = DataGenerator(a, b, mat, batch_size)
    num_samples = len(my_batch_generator) * batch_size
    return my_batch_generator, num_samples



