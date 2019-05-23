from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
import MY_Generator
import keras
import tensorflow as tf
import numpy as np
from os import scandir
import generate_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', 'D:/dataset/train_set/train_set', 'Input directory')
tf.flags.DEFINE_string('test_dir', 'D:/dataset/test_set', 'Test directory')
tf.flags.DEFINE_string('val_dir', 'D:/dataset/val_set', 'Validation directory')

def lr_callback(epochs, lr):
    updated_lr = lr
    # il primo epochs è 0 che ha resto 0
    if ((epochs+1) % 2) == 0:
        updated_lr /= 10
    return updated_lr


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


def train(batch_size, epochs):
    lr = 0.005
    momentum = 0.9

    a_train, b_train, mat_train = data_reader(FLAGS.input_dir)

    a_val, b_val, mat_val = data_reader(FLAGS.val_dir)

    my_training_batch_generator = MY_Generator.Generator(a_train, b_train, mat_train, batch_size)
    my_val_batch_generator = MY_Generator.Generator(a_val, b_val, mat_val, batch_size)
    nn = generate_model.HomographyNN(batch_size=batch_size, epochs=epochs)
    num_training_samples = len(a_train)
    num_validation_samples = len(a_val)
    nn.set_optimizer_sgd(lr=lr, momentum=momentum)
    nn.set_callback(lr_callback)
    nn.build_model()
    nn.compile()
    nn.fit(training_generator=my_training_batch_generator, dimension_train=num_training_samples,
           val_generator=my_val_batch_generator,
           dimension_val=num_validation_samples)
    nn.save_model("my_models_batch128.h5")
    a_test, b_test, mat_test = data_reader(FLAGS.test_dir)
    my_test_batch_generator = MY_Generator.Generator(a_test, b_test, mat_test, batch_size)
    num_test_samples = len(a_test)
    [loss, mtr] = nn.test(test_generator=my_test_batch_generator, dimension_test=num_test_samples)
    return loss, mtr


if __name__ == '__main__':
    loss, mtr = train(128, 6)
    print(loss)
    print(mtr)




    #datagen = ImageDataGenerator()
    # # load and iterate training dataset
    # train_it = datagen.flow_from_directory(FLAGS.input_dir, class_mode=None, batch_size=64)
    # # load and iterate validation dataset
    # val_it = datagen.flow_from_directory(FLAGS.val_dir, class_mode=None, batch_size=64)
    # # load and iterate test dataset
    # test_it = datagen.flow_from_directory(FLAGS.test_dir, class_mode='binary', batch_size=64)
    # model = build_model()
    # num_training_samples = len(a_train)