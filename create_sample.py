import sys
import random
import glob
import os.path
import uuid

from queue import Queue, Empty
from threading import Thread
import glob
import numpy as np
import cv2
import scipy.io as sio

def generate_points():
    # Choose top-left corner of patch (assume 0,0 is top-left of image)
    # Restrict points to within 24-px from the border
    p = 32
    x, y = (random.randint(150, 200), random.randint(150, 200))
    patch = [
        (x, y),
        (x + 128, y),
        (x + 128, y + 128),
        (x, y + 128)
    ]
    # Perturb
    perturbed_patch = [(x + random.randint(-p, p), y + random.randint(-p, p)) for x, y in patch]
    return np.array(patch), np.array(perturbed_patch)


if __name__ == '__main__':

    directory = glob.glob("F:/datasets/ms-coco/resized_train/*.png")
    # Read a image
    for j in range(600000):
        num = np.random.randint(1, len(directory))
        img = cv2.imread(directory[num], cv2.IMREAD_GRAYSCALE)
        # Find the dimension
        img = cv2.resize(img, (480, 480))

        # Find the positions of the points
        points, modified_points = generate_points()

        # Extract the first square
        i_1 = img[points[0, 0]:points[0, 0]+128, points[0, 1]:points[0, 1]+128]


        # Calculate the first homography
        homography, status = cv2.findHomography(points, modified_points)
        homography_value = np.reshape(homography, (9, 1))


        # Transform the first image
        img_modify = cv2.warpPerspective(img, np.linalg.inv(homography), (480, 480))

        # Extract the second image
        a = directory[num].split('\\')[-1].split('.png')[0]
        i_2 = img_modify[points[0, 0]:points[0, 0]+128, points[0, 1]:points[0, 1]+128]
        folder_save = 'D:/tesisti/Boem/ar2/train_set/'
        if i_2.shape == (128, 128):
            cv2.imwrite(folder_save + str(a) + '_' + str(j) + "_A.png", i_1)
            cv2.imwrite(folder_save + str(a) + '_' + str(j) + "_B.png", i_2)
            sio.savemat(folder_save + str(a) + '_' + str(j) + "_M.mat", mdict={"homography": homography_value[0:8]})

        if j % 10000 == 0:
            print(j)