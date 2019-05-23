from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
from keras import optimizers, callbacks


def lr_callback(epochs, lr):
    updated_lr = lr
    # il primo epochs è 0 che ha resto 0
    if ((epochs + 1) % 2) == 0:
        updated_lr /= 10
    return updated_lr


class HomographyNN:

    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.optimizer = None
        self.cb = None

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
        # model1.add(BatchNormalization())

        model1.add(Dropout(0.5))

        model1.add(Dense(8, name='dense_2'))

        model1.summary()
        self.model = model1
        return

    def get_model(self):
        return self.model

    def save_model(self, weights_file):
        """
        Saves the model in a .h5 file
        :param weights_file: name of the file in which to save the weights of the model
        """
        self.model.save_weights(weights_file)
        return

    def load_model(self, weights_path):
        """
        Loads the saved weights in the model
        :param weights_path: path of the file containing the weights
        """
        self.model.load_weights(weights_path)
        return

    def set_optimizer_sgd(self, lr, momentum):
        sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=False)
        self.optimizer = sgd
        return

    def set_callback(self, function):
        callback_learning_rate = callbacks.LearningRateScheduler(function)
        callbacks_early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                                                           mode='min', baseline=None, restore_best_weights=False)
        self.cb = [callback_learning_rate, callbacks_early_stopping]
        return

    def fit(self, training_generator, dimension_train, val_generator, dimension_val):
        self.model.fit_generator(generator=training_generator,
                                 steps_per_epoch=(dimension_train // self.batch_size),
                                 epochs=self.epochs,
                                 verbose=1,
                                 callbacks=self.cb,
                                 validation_data=val_generator,
                                 validation_steps=dimension_val//self.batch_size,
                                 class_weight=None,
                                 max_queue_size=10,
                                 workers=1,
                                 use_multiprocessing=True,
                                 shuffle=True,
                                 initial_epoch=0)
        return

    def test(self, test_generator, dimension_test):
        return self.model.evaluate_generator(generator=test_generator, steps=dimension_test//self.batch_size,
                                             max_queue_size=10, workers=1, use_multiprocessing=True, verbose=1)

    def compile(self,):
        self.model.compile(optimizer=self.optimizer, loss="msle", metrics=["mse"], loss_weights=None,
                           sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
        return
