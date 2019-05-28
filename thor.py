import cv2
import scipy.io
import numpy as np
import copy


class Thor:
    def __init__(self,
                 img_a_name='../ppp.jpg',
                 img_b_name='../pppout.jpg',
                 patch_a_name='../pppA.png',
                 patch_b_name='../pppB.png',
                 h_ab_name='../ppp.mat'
                 ):
        self.img_a = cv2.imread(img_a_name, cv2.IMREAD_GRAYSCALE)
        self.img_b = cv2.imread(img_b_name, cv2.IMREAD_GRAYSCALE)
        self.patch_a = cv2.imread(patch_a_name, cv2.IMREAD_GRAYSCALE)
        self.patch_b = cv2.imread(patch_b_name, cv2.IMREAD_GRAYSCALE)
        mat = scipy.io.loadmat(h_ab_name)['out']
        h = np.append(mat, 1)
        h = np.reshape(h, [3, 3])
        self.h_ab = np.linalg.inv(h)
        self.h_est = None

    def cv_homography_estimate(self):
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.img_a, None)
        kp2, des2 = sift.detectAndCompute(self.img_b, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        h_ab = None
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            h_ab, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.h_est = h_ab
        return h_ab

    def project(self, h, sz=30, color=(0, 255, 0)):
        pt0 = np.array([5, 5, 1]).transpose()
        pt01 = np.matmul(h, pt0)
        pt01 = np.divide(pt01, pt01[2])

        pt1 = np.array([sz, 5, 1]).transpose()
        pt11 = np.matmul(h, pt1)
        pt11 = np.divide(pt11, pt11[2])

        pt2 = np.array([sz, sz, 1]).transpose()
        pt21 = np.matmul(h, pt2)
        pt21 = np.divide(pt21, pt21[2])

        pt3 = np.array([5, sz, 1]).transpose()
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

        pts = np.array([pt0, pt1, pt2, pt3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        out_a = cv2.polylines(cv2.cvtColor(self.img_a, cv2.COLOR_GRAY2BGR), [pts], True, color)

        pts1 = np.array([pt01, pt11, pt21, pt31], np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        out_b = cv2.polylines(cv2.cvtColor(out_a, cv2.COLOR_GRAY2BGR), [pts1], True, color)

        return out_a, out_b

    def project2(self, sz=30):
        color_gt = (0, 255, 0)
        color_est = (0, 255, 255)

        pt0 = np.array([5, 5, 1]).transpose()
        pt01 = np.matmul(self.h_ab, pt0)
        pt01 = np.divide(pt01, pt01[2])
        pt02 = np.matmul(self.h_est, pt0)
        pt02 = np.divide(pt02, pt02[2])

        pt1 = np.array([sz, 5, 1]).transpose()
        pt11 = np.matmul(self.h_ab, pt1)
        pt11 = np.divide(pt11, pt11[2])
        pt12 = np.matmul(self.h_est, pt1)
        pt12 = np.divide(pt12, pt12[2])

        pt2 = np.array([sz, sz, 1]).transpose()
        pt21 = np.matmul(self.h_ab, pt2)
        pt21 = np.divide(pt21, pt21[2])
        pt22 = np.matmul(self.h_est, pt2)
        pt22 = np.divide(pt22, pt22[2])

        pt3 = np.array([5, sz, 1]).transpose()
        pt31 = np.matmul(self.h_ab, pt3)
        pt31 = np.divide(pt31, pt31[2])
        pt32 = np.matmul(self.h_est, pt3)
        pt32 = np.divide(pt32, pt32[2])

        pt0 = np.delete(pt0, 2)
        pt1 = np.delete(pt1, 2)
        pt2 = np.delete(pt2, 2)
        pt3 = np.delete(pt3, 2)

        pt01 = np.delete(pt01, 2)
        pt11 = np.delete(pt11, 2)
        pt21 = np.delete(pt21, 2)
        pt31 = np.delete(pt31, 2)

        pt02 = np.delete(pt02, 2)
        pt12 = np.delete(pt12, 2)
        pt22 = np.delete(pt22, 2)
        pt32 = np.delete(pt32, 2)

        pts = np.array([pt0, pt1, pt2, pt3], np.int32)
        pts = pts.reshape((-1, 1, 2))
        out_a = cv2.polylines(cv2.cvtColor(self.img_a, cv2.COLOR_GRAY2BGR), [pts], True, color_gt)

        pts1 = np.array([pt01, pt11, pt21, pt31], np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        out_b = cv2.polylines(cv2.cvtColor(self.img_b, cv2.COLOR_GRAY2BGR), [pts1], True, color_gt)

        pts2 = np.array([pt02, pt12, pt22, pt32], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))
        out_b = cv2.polylines(out_b, [pts2], True, color_est)

        return out_a, out_b
