from thor import Thor
import cv2

nm = 'COCO_test2014_000000000014'
test_base = '../coco_test/' + nm
img_a = test_base + '.jpg'
img_b = None
patch_a = test_base + '_1_1.jpg'
patch_b = test_base + '_1_2.jpg'
perts = test_base + '_1_re.txt'
pert_corners = test_base + '_1_ab.txt'
model_weights1 = 'weights/mse_weights00000020.h5'
model_weights2 = 'weights/mse_adam_weights00000010.h5'

t = Thor(img_a, img_b, patch_a, patch_b, perts, pert_corners)
size = 100
x0 = 150
y0 = 150

# Ground truth
h_gt = t.h_ab
color_gt = [0, 255, 0]

# SIFT
h_sift = t.cv_sift_homography_estimate()
color_sift = [255, 0, 0]

# ORB
h_orb = t.cv_orb_homography_estimate()
color_orb = [0, 255, 255]

# SGD
h_cnn = t.cnn_4points_estimate(model_weights1)
color_cnn = [0, 0, 255]

# Adam
h_cnn2 = t.cnn_4points_estimate(model_weights2)
color_cnn2 = [255, 0, 255]

h = [h_gt, h_sift, h_cnn, h_cnn2]
colors = [color_gt, color_sift, color_cnn, color_cnn2]
names = ['', 'SIFT', 'CNN SGD', 'CNN Adam', 'ORB']

out_a, out_b = t.multiple_project(h, colors, x0, y0, size, names)

cv2.namedWindow('homographies', cv2.WINDOW_GUI_EXPANDED)
cv2.imshow('homographies', out_b)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('../test_data/'+nm+'_0.jpg', out_a)
cv2.imwrite('../test_data/'+nm+'_1.jpg', out_b)









