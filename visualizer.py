from thor import Thor
import numpy as np
import cv2

test_folder = '../test_data/test1/'
img_a = test_folder + 'COCO_train2014_000000000009.jpg'
img_b = None
patch_a = test_folder + 'COCO_train2014_000000000009_1_1.jpg'
patch_b = test_folder + 'COCO_train2014_000000000009_1_2.jpg'
perts = test_folder + 'COCO_train2014_000000000009_1_re.txt'
pert_corners = test_folder + 'COCO_train2014_000000000009_1_ab.txt'
model_weights = 're_12ep_msle'

t = Thor(img_a, img_b, patch_a, patch_b, perts, pert_corners)
size = 300
x0 = 50
y0 = 50

h_gt = t.h_ab
color_gt = [0, 255, 0]

h_sift = t.cv_sift_homography_estimate()
color_sift = [255, 0, 0]

h_orb = t.cv_orb_homography_estimate()
color_orb = [0, 255, 255]

h_cnn = t.cnn_4points_estimate(model_weights)
color_cnn = [0, 0, 255]

h = [h_gt, h_sift, h_orb, h_cnn]
colors = [color_gt, color_sift, color_orb, color_cnn]

out_a, out_b = t.multiple_project(h, colors, x0, y0, size)

cv2.namedWindow('homographies', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('homographies', out_b)
cv2.waitKey(0)
cv2.destroyAllWindows()




'''
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


im_a1, im_b1 = t.project2(sz)
i1 = cv2.hconcat([im_a1, im_b1])

cv2.namedWindow('homographies', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('homographies', i1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''








