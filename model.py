from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
import numpy as np
import keras


class HomographyNet:

    def __init__(self, name, batch_size):
        self.name = name
        self.model = None
        self.batch_size = batch_size

    def build_model(self):
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
        model1.add(InputLayer(input_shape=(128, 128, 2), batch_size=self.batch_size, name='input_layer'))

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
        #model1.add(BatchNormalization())

        model1.add(Dropout(0.5))

        model1.add(Dense(8, name='dense_2'))

        model1.summary()
        self.model = model1

    def train_model(self, loss, base_lr, momentum, train_data, val_data=None, epochs=1, steps_per_epoch=500, val_steps=5, test_data=None):
        """
        Compiles and trains the model according to the given parameters.
        Optimizer: Stochastic Gradient Descent on MSE loss
        :return: loss and accuracy on test set
        """
        sgd = keras.optimizers.SGD(lr=base_lr, momentum=momentum)
        self.model.compile(loss=loss, optimizer=sgd, metrics=['msle'])

        def _lr_callback(epoch, lr):
            updated_lr = lr
            if (epoch % 2) == 0:
                updated_lr /= 10
            return updated_lr

        callback_learning_rate = keras.callbacks.LearningRateScheduler(_lr_callback)
        callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        callbacks = [callback_learning_rate, callback_early_stopping]

        self.model.fit(x=keras.Input(tensor=train_data[0]),
                       y=keras.Input(tensor=train_data[1]),
                       epochs=epochs,
                       steps_per_epoch=steps_per_epoch,
                       verbose=1,
                       callbacks=callbacks,
                       validation_data=val_data,
                       validation_steps=val_steps)

        [loss, mtr] = self.model.evaluate(x=test_data[0], y=test_data[1], steps=5, verbose=1)
        return loss, mtr

    def save_model(self, weights_file):
        """
        Saves the model in a .h5 file
        :param weights_file: name of the file in which to save the weights of the model
        """
        self.model.save_weights(weights_file)

    def load_model(self, weights_path):
        """
        Loads the saved weights in the model
        :param weights_path: path of the file containing the weights
        """
        self.model.load_weights(weights_path)














