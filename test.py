import cv2
import scipy.io
import numpy as np

name_in = 'prova/input.png'
name_out = 'prova/output.png'
name_sq1 = 'prova/train_000002A.png'
name_sq2 = 'prova/train_000002B.png'
name_h = 'prova/train_000002.mat'

img_in = cv2.imread(name_in, cv2.IMREAD_GRAYSCALE)
img_out = cv2.imread(name_out, cv2.IMREAD_GRAYSCALE)

sq1 = cv2.imread(name_sq1, cv2.IMREAD_GRAYSCALE)
sq2 = cv2.imread(name_sq2, cv2.IMREAD_GRAYSCALE)

mat = scipy.io.loadmat(name_h)['out']
h = np.append(mat, 1)
h = np.reshape(h, [3, 3])
h = np.linalg.inv(h)

pt0 = np.array([5, 5, 1]).transpose()
pt01 = np.matmul(h, pt0)
pt01 = np.divide(pt01, pt01[2])

pt1 = np.array([35, 5, 1]).transpose()
pt11 = np.matmul(h, pt1)
pt11 = np.divide(pt11, pt11[2])

pt2 = np.array([35, 35, 1]).transpose()
pt21 = np.matmul(h, pt2)
pt21 = np.divide(pt21, pt21[2])

pt3 = np.array([5, 35, 1]).transpose()
pt31 = np.matmul(h, pt3)
pt31 = np.divide(pt31, pt31[2])

pt0 = np.delete(pt0, 2)
pt1 = np.delete(pt1, 2)
pt2 = np.delete(pt2, 2)
pt3 = np.delete(pt3, 2)

pt01 = np.delete(pt01, 2)
pt11 = np.delete(pt11, 2)
pt21 = np.delete(pt21, 2)
pt31 = np.delete(pt31, 2)

########################################################################################################################
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(sq1, None)
kp2, des2 = sift.detectAndCompute(sq2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
M=0
if len(good)>5:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
########################################################################################################################
pts = np.array([pt0, pt1, pt2, pt3], np.int32)
pts = pts.reshape((-1, 1, 2))
pp = cv2.polylines(img_in, [pts], True, (0, 255, 255))

pts1 = np.array([pt01, pt11, pt21, pt31], np.int32)
pts1 = pts1.reshape((-1, 1, 2))
pp1 = cv2.polylines(img_out, [pts1], True, (0, 255, 255))

mse = np.sum((np.linalg.inv(h)-M)**2)
print(mse)

'''
cv2.imshow('image', pp1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




k= 4