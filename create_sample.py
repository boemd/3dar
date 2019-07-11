
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

    directory = glob.glob("D:/Downloads/unlabeled2017/*.jpg")
    # Read a image
    for j in range(25000):
        num = np.random.randint(1, 287000)
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
        a = directory[num].split('\\')[-1].split('.jpg')[0]
        i_2 = img_modify[points[0, 0]:points[0, 0]+128, points[0, 1]:points[0, 1]+128]
        if i_2.shape == (128, 128):
            cv2.imwrite("D:/Downloads/val_set/" + str(a) + '_' + str(j) + "_A.png", i_1)
            cv2.imwrite("D:/Downloads/val_set/" + str(a) + '_' + str(j) + "_B.png", i_2)
            sio.savemat("D:/Downloads/val_set/" + str(a) + '_' + str(j) + "_M.mat", mdict={"homography": homography_value[0:8]})

        if j % 10000 == 0:
            print(j)

#
#
#
# def homography(points):
#     x1, y1 = points[0]
#     x2, y2 = points[1]
#     x3, y3 = points[2]
#     x4, y4 = points[3]
#     xp1, yp1 = points[4]
#     xp2, yp2 = points[5]
#     xp3, yp3 = points[6]
#     xp4, yp4 = points[7]
#
#     A = [[-x1, -y1, -1, 0, 0, 0,  x1 * xp1, y1 * xp1, xp1],
#          [0, 0, 0, - x1, - y1, - 1, x1 * yp1, y1 * yp1, yp1],
#          [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
#          [0, 0, 0, - x2, - y2, - 1, x2 * yp2, y2 * yp2, yp2],
#          [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
#          [0, 0, 0, - x3, - y3, - 1, x3 * yp3, y3 * yp3, yp3],
#          [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y1 * xp4, xp4],
#          [0, 0, 0, - x4, - y4, - 1, x4 * yp4, y4 * yp4, yp4]]
#
#     [U, S, V] = np.linalg.svd(A)
#
#     H = V[:, -1]
#     H = np.reshape(H, (3, 3))
#     return H
