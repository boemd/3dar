
import numpy as np

import sklearn.metrics as sk


def mse(true, predicted):
    return sk.mean_squared_error(true.T, predicted.T, multioutput='raw_values')


def mean_mse(true, predicted):
    error = sk.mean_squared_error(true.T, predicted.T, multioutput='raw_values')
    mean = np.mean(error)
    return mean


if __name__ == '__main__':
    a = np.array([[4, 3], [2, 3], [5, 4]])
    b = np.array([[2, 3], [2, 4], [6, 5]])
    t = mse(a, b)
    print(t)
    t_mean = mean_mse(a, b)
    print(t_mean)