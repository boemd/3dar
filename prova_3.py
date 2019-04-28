from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
import matplotlib.image as img
import MY_Generator
import glob

import numpy as np
import logging
import keras
import tensorflow as tf
import os
import scipy.io as sio
import random
from os import scandir

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', 'C:/Users/gabro/Documents/3D/Dataset/db_test_set', 'Input directory')
tf.flags.DEFINE_string('test_dir', 'C:/Users/gabro/Documents/3D/Dataset/db_test_set', 'Test directory')
tf.flags.DEFINE_string('val_dir', 'C:/Users/gabro/Documents/3D/Dataset/db_test_set', 'Validation directory')


def build_model():
    """
    Builds a Sequential model by stacking:
        Input layer
        4 conv layers (num filters: 64, kernel size: 3) with batch normalization and relu activation
        4 conv layers (num filters: 128, kernel size: 3) with batch normalization and relu activation
        Max-Pooling (2x2) is performed every 2 convolutions
        Fully connected layer (depth: 1024)
        Output layer
    :return: the built model
    """
    model1 = Sequential()
    model1.add(InputLayer(input_shape=(128, 128,1), name='input_layer'))

    model1.add(Conv2D(64, 3, name='conv_64_1'))
    model1.add(BatchNormalization(name='batch_norm_1'))
    model1.add(Activation('relu', name='relu_1'))

    model1.add(Conv2D(64, (3, 3), name='conv_64_2'))
    model1.add(BatchNormalization(name='batch_norm_2'))
    model1.add(Activation('relu', name='relu_2'))

    model1.add(MaxPooling2D(pool_size=(2, 2), name='pool_1'))

    model1.add(Conv2D(64, (3, 3), name='conv_64_3'))
    model1.add(BatchNormalization(name='batch_norm_3'))
    model1.add(Activation('relu', name='relu_3'))

    model1.add(Conv2D(64, (3, 3), name='conv_64_4'))
    model1.add(BatchNormalization(name='batch_norm_4'))
    model1.add(Activation('relu', name='relu_4'))

    model1.add(MaxPooling2D(pool_size=(2, 2), name='pool_2'))

    model1.add(Conv2D(128, (3, 3), name='conv_128_1'))
    model1.add(BatchNormalization(name='batch_norm_5'))
    model1.add(Activation('relu', name='relu_5'))

    model1.add(Conv2D(128, (3, 3), name='conv_128_2'))
    model1.add(BatchNormalization(name='batch_norm_6'))
    model1.add(Activation('relu', name='relu_6'))

    model1.add(MaxPooling2D(pool_size=(2, 2), name='pool_3'))

    model1.add(Conv2D(128, (3, 3), name='conv_128_3'))
    model1.add(BatchNormalization(name='batch_norm_7'))
    model1.add(Activation('relu', name='relu_7'))

    model1.add(Conv2D(128, (3, 3), name='conv_128_4'))
    model1.add(BatchNormalization(name='batch_norm_8'))
    model1.add(Activation('relu', name='relu_8'))

    # model1.add(MaxPooling2D(pool_size=(2, 2)))

    model1.add(Flatten(name='flatten'))

    model1.add(Dropout(0.5))

    model1.add(Dense(1024, name='dense_1'))
    model1.add(Activation('relu', name='relu_9'))
    # model1.add(BatchNormalization())

    model1.add(Dropout(0.5))

    model1.add(Dense(8, name='dense_2'))
    model = model1
    model1.summary()

    return model


def _lr_callback(epochs, lr):
    updated_lr = lr
    #il primo epochs è 0 che ha resto 0
    if ((epochs+1) % 2) == 0:
        updated_lr /= 10
    return updated_lr


def generator(features_1, features_2, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#

     batch_features = np.zeros((batch_size, 128, 128, 2))
     batch_labels = np.zeros((batch_size, 1))

     while True:
       for i in range(batch_size):
         # choose random index in features
         index= random.choice(len(features_1), 1)
         batch_features[i] =[features_1[index],features_2[index]]
         batch_labels[i] = labels[index]
       yield batch_features, batch_labels


def train(batch_size, epochs):
    lr = 0.005
    momentum = 0.9
    x_train_1 = np.load('E:/x_train_1.npy')
    print(1)
    x_train_2 = np.load('E:/x_train_2.npy')
    print(2)
    y_train = np.load('E:/y_train.npy')

    lr = 0.005
    momentum = 0.9
    callback_learning_rate = keras.callbacks.LearningRateScheduler(_lr_callback)
    callbacks_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                                                             mode='min', baseline=None, restore_best_weights=False)
    model = build_model()
    sgd = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss="msle", metrics=["mse"], loss_weights=None,
                  sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

    model.fit_generator(generator(x_train_1, x_train_2, y_train, batch_size),
                        verbose=1,
                        callbacks=None,
                        validation_data=None,
                        validation_steps=None,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=True,
                        shuffle=True,
                        initial_epoch=1)
    #validation_steps=None,
    #validation_freq=1)
    # a_test, b_test, mat_test = data_reader(FLAGS.test_dir)
    # x_test = image_matrix_creation(a_test, b_test)
    # y_test = mat_matrix_creation(mat_test)
    # [loss, mtr] = model.evaluate(x=x_test, y=y_test, batch_size=None, verbose=1, sample_weight=None, steps=None,)
    # print(loss, mtr)
    return


if __name__ == '__main__':
    train(32, 10)



    #datagen = ImageDataGenerator()
    # # load and iterate training dataset
    # train_it = datagen.flow_from_directory(FLAGS.input_dir, class_mode=None, batch_size=64)
    # # load and iterate validation dataset
    # val_it = datagen.flow_from_directory(FLAGS.val_dir, class_mode=None, batch_size=64)
    # # load and iterate test dataset
    # test_it = datagen.flow_from_directory(FLAGS.test_dir, class_mode='binary', batch_size=64)
    # model = build_model()
    # num_training_samples = len(a_train)