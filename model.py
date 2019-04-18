from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout

class HomographyNet:

    def __init__(self, name):
        self.name = name


    def build_model(self):
        model1 = Sequential()
        model1.add(Conv2D(64, (3, 3), input_shape=(128, 128, 2)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Conv2D(64, (3, 3), input_shape=(128, 128, 2)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(MaxPooling2D(pool_size=(2, 2)))

        model1.add(Conv2D(64, (3, 3), input_shape=(128, 128, 2)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Conv2D(128, (3, 3), input_shape=(128, 128, 2)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(MaxPooling2D(pool_size=(2, 2)))

        model1.add(Conv2D(128, (3, 3), input_shape=(128, 128, 2)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Conv2D(128, (3, 3), input_shape=(128, 128, 2)))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Flatten())
        model1.add(Dropout(0.5))

        model1.add(Dense(1024, activation='relu'))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Dense(1025, activation='relu'))
        model1.add(BatchNormalization())
        model1.add(Activation('relu'))

        model1.add(Dense(8))

        return model1






