from thor import Thor
import numpy as np
import cv2

t = Thor()
sz = 250
h = t.cv_homography_estimate()
t.cnn_homography_estimate('models/batch128')

im_a1, im_b1 = t.project(t.h_ab, sz, (0, 255, 0))
i1 = cv2.hconcat([im_a1, im_b1])

cv2.namedWindow('ground truth', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('ground truth', i1)
cv2.waitKey(0)
cv2.destroyAllWindows()

im_a2, im_b2 = t.project(t.h_homo, sz, (0, 255, 255))
i2 = cv2.hconcat([im_a2, im_b2])

cv2.namedWindow('patch estimation', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('patch estimation', i2)
cv2.waitKey(0)
cv2.destroyAllWindows()

mse = np.sum((t.h_homo - t.h_ab)**2)
print(mse)

'''
im_a1, im_b1 = t.project2(sz)
i1 = cv2.hconcat([im_a1, im_b1])

cv2.namedWindow('homographies', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('homographies', i1)
cv2.waitKey(0)
cv2.destroyAllWindows()

mse = np.sum((t.h_est - t.h_ab)**2)
print(mse)
'''







