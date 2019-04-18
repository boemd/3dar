from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
import keras
import h5py

class HomographyNet:

    def __init__(self, name, batch_size):
        self.name = name
        self.model = None
        self.batch_size = batch_size

    def build_model(self):
        model1 = Sequential()
        model1.add(InputLayer(batch_input_shape=(self.batch_size, 128, 128, 2)))
        model1.add(Conv2D(64, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Conv2D(64, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(MaxPooling2D(pool_size=(2, 2)))

        model1.add(Conv2D(64, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Conv2D(64, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(MaxPooling2D(pool_size=(2, 2)))

        model1.add(Conv2D(128, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Conv2D(128, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(MaxPooling2D(pool_size=(2, 2)))

        model1.add(Conv2D(128, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Conv2D(128, (3, 3)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        # model1.add(MaxPooling2D(pool_size=(2, 2)))

        model1.add(Flatten())

        model1.add(Dropout(0.5))

        model1.add(Dense(1024, activation='relu'))
        #model1.add(BatchNormalization())

        model1.add(Dropout(0.5))

        model1.add(Dense(8))

        self.model = model1

    def train_model(self, base_lr, momentum, train_data, val_data, epochs):

        def _lr_callback(self, epoch, lr):
            updated_lr = lr
            if (epoch % 4) == 0:
                updated_lr /= 10
            return updated_lr

        callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        #callback_learning_rate = keras.callbacks.LearningRateScheduler(_lr_callback)
        callbacks = [callback_early_stopping]#, callback_learning_rate]
        sgd = keras.optimizers.SGD(lr=base_lr, momentum=momentum)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        self.model.fit_generator(generator=train_data,
                                 epochs=epochs,
                                 steps_per_epoch=7813,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=val_data)
                                 #validation_freq=2)

    def save_model(self, weights_file):
        self.model.save_weights(weights_file)

    def load_model(self, weights_path):
        self.model.load_weights(weights_path)














