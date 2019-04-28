from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
import matplotlib.image as img
import glob
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import logging
import keras
import tensorflow as tf
import os
import scipy.io as sio
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
    model1.add(InputLayer(input_shape=(128, 128, 1), name='input_layer'))

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


def data_loader(path, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        for npz in glob.glob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            # Yield minibatch
            for i in range(0, len(offsets), batch_size):
                end_i = i + batch_size
                try:
                    batch_images = images[i:end_i]
                    batch_offsets = offsets[i:end_i]
                except IndexError:
                    continue
                # Normalize
                batch_images = (batch_images - 127.5) / 127.5
                batch_offsets = batch_offsets / 32.
                yield batch_images, batch_offsets

def train(batch_size, epochs):
    # Dataset-specific
    train_data_path = FLAGS.input_dir
    test_data_path = FLAGS.test_dir
    lr = 0.005
    momentum = 0.9
    filepath = "checkpoints/weights-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
    callback_list = [checkpoint]
    model = build_model()
    sgd = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss="msle", metrics=["mse"], loss_weights=None,
                  sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    # Train
    print('TRAINING...')
    model.fit_generator(data_loader(train_data_path, batch_size),
                        steps_per_epoch=32,
                        epochs=epochs, callbacks=callback_list,verbose=1)


if __name__ == '__main__':
        train(32, 10)
