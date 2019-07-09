from skimage.io import imread
import numpy as np
import scipy.io as sio


class Generator:

    def __init__(self, image_filenames_1, image_filenames_2, labels, batch_size):
        self.image_filenames_1, self.image_filenames_2, self.labels = image_filenames_1, image_filenames_2, labels
        self.batch_size = batch_size
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __len__(self):
        return np.int(np.ceil(len(self.image_filenames_1) / float(self.batch_size)))

    def __next__(self):
        batch_x_1 = self.image_filenames_1[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]
        batch_x_2 = self.image_filenames_2[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]
        batch_y = self.labels[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]
        batch_x_train = np.zeros((len(batch_x_1), 128, 128, 2))
        batch_x_train[:, :, :, 0] = np.array([imread(file_name_1) for file_name_1 in batch_x_1])
        batch_x_train[:, :, :, 1] = np.array([imread(file_name_2) for file_name_2 in batch_x_2])
        batch_y_train = np.zeros((len(batch_y), 8))
        for i in range(len(batch_y)):

             batch_y_train[i, :] = sio.loadmat(self.labels[i])['out'].T

        batch_x_train = np.divide(batch_x_train, 255)
        batch_y_train = np.divide(batch_y_train, 10)
        return batch_x_train, batch_y_train

