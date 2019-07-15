import numpy as np
from skimage.io import imread
import keras
import scipy.io as sio


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, image_filenames_0, image_filenames_1, labels, batch_size):
        self.image_filenames_0, self.image_filenames_1, self.labels = image_filenames_0, image_filenames_1, labels
        self.batch_size = batch_size
        # self.idx = 0
        self.is_mat = self.labels[0].endswith('mat')
        self.max_idx = len(labels)//batch_size

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return self.max_idx

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        batch_x_0_names = self.image_filenames_0[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x_1_names = self.image_filenames_1[index * self.batch_size: (index + 1) * self.batch_size]
        batch_y_names = self.labels[index * self.batch_size: (index + 1) * self.batch_size]

        batch_x = np.zeros((self.batch_size, 128, 128, 2))
        batch_x[:, :, :, 0] = np.array([imread(file_name) for file_name in batch_x_0_names])
        batch_x[:, :, :, 1] = np.array([imread(file_name) for file_name in batch_x_1_names])

        batch_y = np.zeros((self.batch_size, 8))
        if self.is_mat:
            for i in range(self.batch_size):
                batch_y[i, :] = sio.loadmat(batch_y_names[i])['homography'].T
        else:
            for i in range(self.batch_size):
                file = open(batch_y_names[i])
                lines = file.readlines()
                file.close()
                ts = []
                for line in lines:
                    ts.append(float(line[:-1]))
                batch_y[i, :] = np.asarray(ts)

        # Rescale the data to feed to the CNN
        batch_x = np.divide(np.subtract(batch_x, 127.5), 127.5)
        batch_y = np.divide(batch_y, 32)

        return batch_x, batch_y

